#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dazel: A tool to execute Bazel commands in a Docker container."""

import getpass
import hashlib
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import collections
from typing import Any, Dict, List


DAZEL_RC_FILE = ".dazelrc"
DAZEL_RUN_FILE = ".dazel_run"
BAZEL_MODULE_FILE = "MODULE.bazel"

DEFAULT_INSTANCE_NAME = "dazel"
DEFAULT_IMAGE_NAME = "dazel"
DEFAULT_RUN_COMMAND = "/bin/bash"
DEFAULT_DOCKER_COMMAND = "docker"
DEFAULT_LOCAL_DOCKERFILE = "Dockerfile.dazel"
DEFAULT_REMOTE_REPOSITORY = "dazel"
DEFAULT_DIRECTORY = os.getcwd()
DEFAULT_COMMAND = "/usr/bin/bazel"
DEFAULT_VOLUMES = []
DEFAULT_PORTS = []
DEFAULT_ENV_VARS = []
DEFAULT_GPUS = []
DEFAULT_PLATFORM = ""
DEFAULT_SHM_SIZE = ""
DEFAULT_NETWORK = "dazel"
DEFAULT_RUN_DEPS = []
DEFAULT_DOCKER_COMPOSE_FILE = ""
DEFAULT_DOCKER_COMPOSE_COMMAND = "docker-compose"
DEFAULT_DOCKER_COMPOSE_PROJECT_NAME = "dazel"
DEFAULT_DOCKER_COMPOSE_SERVICES = ""
DEFAULT_USER = ""
DEFAULT_DOCKER_BUILD_ARGS = ""
DEFAULT_IMAGE_TAG = ""
DEFAULT_DAZEL_FILES_WATCH = []

DEFAULT_DELEGATED_VOLUME = True
DEFAULT_BAZEL_USER_OUTPUT_ROOT = os.path.expanduser("~/.cache/bazel/_bazel_%s" % getpass.getuser())
TEMP_BAZEL_OUTPUT_USER_ROOT = "/var/bazel/workspace/_bazel_%s" % getpass.getuser()
DEFAULT_BAZEL_USER_OUTPUT_PATHS = ["external", "action_cache", "execroot"]
DEFAULT_BAZEL_RC_FILE = ""
DEFAULT_DOCKER_RUN_PRIVILEGED = False
DEFAULT_DOCKER_MACHINE = None
DEFAULT_WORKSPACE_HEX = False

DOCKER_SPECIAL_NETWORK_NAMES = ["host", "bridge", "none"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("dazel")


class DockerInstance:
    """Manages communication and runs commands on associated docker container.

    A DockerInstance can build the image for the container if necessary, run it,
    set it up through configuration variables, and pass on commands to it.
    It streams the output directly and blocks until the command finishes.
    """

    def __init__(
        self,
        instance_name,
        image_name,
        run_command,
        docker_command,
        dockerfile,
        repository,
        directory,
        command,
        volumes,
        ports,
        env_vars,
        gpus,
        network,
        run_deps,
        docker_compose_file,
        docker_compose_command,
        docker_compose_project_name,
        docker_compose_services,
        bazel_user_output_root,
        bazel_rc_file,
        docker_run_privileged,
        docker_machine,
        dazel_run_file,
        workspace_hex,
        delegated_volume,
        user,
        docker_build_args,
        shm_size,
        platform,
        image_tag,
        files_watch,
    ):
        real_directory = os.path.realpath(directory)
        self.workspace_hex_digest = ""
        self.instance_name = instance_name
        self.image_name = image_name
        self.run_command = run_command
        self.docker_command = docker_command
        self.dockerfile = dockerfile
        self.repository = repository
        self.directory = directory
        self.command = command
        self.network = network
        self.docker_compose_file = docker_compose_file
        self.docker_compose_command = docker_compose_command
        self.docker_compose_project_name = docker_compose_project_name
        self.bazel_user_output_root = bazel_user_output_root
        self.bazel_output_base = ""
        self.bazel_rc_file = bazel_rc_file
        self.docker_run_privileged = docker_run_privileged
        self.docker_machine = docker_machine
        self.dazel_run_file = dazel_run_file
        self.delegated_volume_flag = ":delegated" if delegated_volume else ""
        self.user = user
        self.platform = platform
        self.docker_build_args = docker_build_args
        self.shm_size = shm_size
        self.remote_directory = self._get_remote_directory(real_directory, add_drive=True)
        self.image_tag = image_tag
        self.files_watch = files_watch

        if workspace_hex:
            self.workspace_hex_digest = hashlib.md5(real_directory.encode("ascii")).hexdigest()
            self.instance_name = "%s_%s" % (self.instance_name, self.workspace_hex_digest)
            self.docker_compose_project_name = "%s%s" % (self.docker_compose_project_name, self.workspace_hex_digest)
            if os.path.exists(self.dockerfile):
                self.image_name = "%s_%s" % (self.image_name, self.workspace_hex_digest)

        if not self.image_tag:
            # Hash the files that may affect the result of the "docker build" command,
            # and use that as the image tag. The image tag will end up as part of the
            # "docker run" command, and hence anything that could possibly change the
            # result of the "docker build" command will be detected and a new image
            # will be built.
            self.image_tag = self._hash_files(sorted(self.files_watch))

        if self.docker_compose_file and not self._is_predefined_network():
            self.network = "%s_%s" % (self.docker_compose_project_name, network)

        self._add_volumes(volumes)
        self._add_ports(ports)
        self._add_env_vars(env_vars)
        self._add_gpus(gpus)
        self._add_run_deps(run_deps)
        self._add_compose_services(docker_compose_services)

    @staticmethod
    def _hash_files(files) -> str:
        """Compute a hash of the contents of files.

        Note that the order of the files affects the hash.

        Args:
            files (list of str): List of paths to files to hash.

        Returns:
            Hash computed from the tracked files.
        """
        m = hashlib.md5()
        for filename in files:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    m.update(f.read())

        return m.hexdigest()

    @classmethod
    def from_config(cls):
        config = cls._config_from_files()
        config.update(cls._config_from_environment())
        return DockerInstance(
            instance_name=config.get("DAZEL_INSTANCE_NAME", DEFAULT_INSTANCE_NAME),
            image_name=config.get("DAZEL_IMAGE_NAME", DEFAULT_IMAGE_NAME),
            run_command=config.get("DAZEL_RUN_COMMAND", DEFAULT_RUN_COMMAND),
            docker_command=config.get("DAZEL_DOCKER_COMMAND", DEFAULT_DOCKER_COMMAND),
            dockerfile=config.get("DAZEL_DOCKERFILE", DEFAULT_LOCAL_DOCKERFILE),
            repository=config.get("DAZEL_REPOSITORY", DEFAULT_REMOTE_REPOSITORY),
            directory=config.get("DAZEL_DIRECTORY", DEFAULT_DIRECTORY),
            command=config.get("DAZEL_COMMAND", DEFAULT_COMMAND),
            volumes=config.get("DAZEL_VOLUMES", DEFAULT_VOLUMES),
            ports=config.get("DAZEL_PORTS", DEFAULT_PORTS),
            env_vars=config.get("DAZEL_ENV_VARS", DEFAULT_ENV_VARS),
            gpus=config.get("DAZEL_GPUS", DEFAULT_GPUS),
            platform=config.get("DAZEL_PLATFORM", DEFAULT_PLATFORM),
            shm_size=config.get("DAZEL_SHM_SIZE", DEFAULT_SHM_SIZE),
            network=config.get("DAZEL_NETWORK", DEFAULT_NETWORK),
            run_deps=config.get("DAZEL_RUN_DEPS", DEFAULT_RUN_DEPS),
            docker_compose_file=config.get("DAZEL_DOCKER_COMPOSE_FILE", DEFAULT_DOCKER_COMPOSE_FILE),
            docker_compose_command=config.get("DAZEL_DOCKER_COMPOSE_COMMAND", DEFAULT_DOCKER_COMPOSE_COMMAND),
            docker_compose_project_name=config.get(
                "DAZEL_DOCKER_COMPOSE_PROJECT_NAME", DEFAULT_DOCKER_COMPOSE_PROJECT_NAME
            ),
            docker_compose_services=config.get("DAZEL_DOCKER_COMPOSE_SERVICES", DEFAULT_DOCKER_COMPOSE_SERVICES),
            bazel_rc_file=config.get("DAZEL_BAZEL_RC_FILE", DEFAULT_BAZEL_RC_FILE),
            bazel_user_output_root=config.get("DAZEL_BAZEL_USER_OUTPUT_ROOT", DEFAULT_BAZEL_USER_OUTPUT_ROOT),
            docker_run_privileged=config.get("DAZEL_DOCKER_RUN_PRIVILEGED", DEFAULT_DOCKER_RUN_PRIVILEGED),
            docker_machine=config.get("DAZEL_DOCKER_MACHINE", DEFAULT_DOCKER_MACHINE),
            dazel_run_file=config.get("DAZEL_RUN_FILE", DAZEL_RUN_FILE),
            workspace_hex=config.get("DAZEL_WORKSPACE_HEX", DEFAULT_WORKSPACE_HEX),
            delegated_volume=config.get("DAZEL_DELEGATED_VOLUME", DEFAULT_DELEGATED_VOLUME),
            user=config.get("DAZEL_USER", DEFAULT_USER),
            docker_build_args=config.get("DAZEL_DOCKER_BUILD_ARGS", DEFAULT_DOCKER_BUILD_ARGS),
            image_tag=config.get("DAZEL_IMAGE_TAG", DEFAULT_IMAGE_TAG),
            files_watch=config.get("DAZEL_FILES_WATCH", DEFAULT_DAZEL_FILES_WATCH),
        )

    def send_command(self, args, quiet=False):
        term_size = shutil.get_terminal_size()

        docker_exec_command = "%s exec -i -e COLUMNS=%s -e LINES=%s -e TERM=%s %s %s %s %s %s" % (
            self.docker_command,
            term_size.columns,
            term_size.lines,
            os.environ.get("TERM", ""),
            self.env_vars,  # Pass DAZEL_ENV_VARS to docker exec
            "-t" if sys.stdout.isatty() else "",
            "--privileged" if self.docker_run_privileged else "",
            ("--user=%s" % self.user if self.user else ""),
            self.instance_name,
        )

        if not self.user:
            output_args = (
                "--output_user_root=%s --output_base=%s" % (TEMP_BAZEL_OUTPUT_USER_ROOT, self.bazel_output_base)
                if self.command and self.bazel_output_base
                else ""
            )
        else:
            output_args = (
                "--output_user_root=%s" % (self.bazel_user_output_root)
                if self.command and self.bazel_user_output_root
                else ""
            )

        command = "%s %s %s %s %s" % (
            docker_exec_command,
            self.command,
            ("--bazelrc=%s" % self.bazel_rc_file if self.bazel_rc_file and self.command else ""),
            output_args,
            '"%s"' % '" "'.join(args),
        )

        command = self._with_docker_machine(command)
        if quiet:
            command += " > /dev/null 2>&1"
        rc = os.system(command)

        if sys.platform == "win32":
            self._fix_win_symlink(docker_exec_command)
            return rc
        else:
            return os.WEXITSTATUS(rc)

    def _fix_win_symlink(self, docker_exec_command):
        p = pathlib.Path(self.directory)
        for path in list(p.glob("bazel-*")):
            command = "%s realpath %s" % (docker_exec_command, str(path.name))
            try:
                output = self._run_command(command).strip()
            except subprocess.CalledProcessError:
                logger.info("INFO: Skipping fixing symlink, it already exists.")
            else:
                drive = pathlib.PureWindowsPath(path.drive)
                local_directory = drive.joinpath(pathlib.PurePosixPath(output))
                path.unlink()
                path.symlink_to(local_directory, target_is_directory=True)

    def start(self):
        """Starts the dazel docker container."""
        rc = 0

        # Verify that the docker executable exists.
        if not self._docker_exists():
            logger.error("ERROR: Docker executable could not be found!")
            return 1

        # Build or pull the relevant dazel image.
        if os.path.exists(self.dockerfile):
            rc = self._build()
        else:
            rc = self._pull()
            # If we have the image, don't stop everything just because we
            # couldn't pull.
            if rc and self._image_exists():
                rc = 0
        if rc:
            return rc

        # If given a docker-compose file, start the services needed to run.
        if self.docker_compose_file and self._docker_compose_exists():
            rc = self._start_compose_services()
        else:
            # If not through docker-compose, run the various dependencies as
            # necessary ourselves.

            # Setup the network if necessary.
            if not self._network_exists() and not self._is_predefined_network():
                logger.info("Creating network: '%s'" % self.network)
                rc = self._start_network()
            if rc:
                return rc

            # Setup run dependencies if necessary.
            rc = self._start_run_deps()
        if rc:
            return rc

        # Run the container itself.
        return self._run_container()

    def is_running(self):
        """Checks if the container is currently running."""
        command = self._with_docker_machine(
            "%s ps  --no-trunc --filter name=^%s$" % (self.docker_command, self.instance_name)
        )
        output = self._run_command(command)
        is_running = self._string_exists(self.instance_name, output)

        # Check if the image tag of the running container differs
        if is_running:
            command = self._with_docker_machine(
                "%s inspect --format='{{.Config.Image}}' %s" % (self.docker_command, self.instance_name)
            )

            output = self._run_command(command)
            image_tag = output.split(":")[-1].strip()
            is_running = image_tag == self.image_tag

        # If we have a directory, make sure the running container is mapped to
        # the same one (if not we need to create a new container mapped to the
        # correct folder).
        if self.directory and is_running:
            real_directory = os.path.realpath(self.directory)
            dir_string = "%s:%s" % (real_directory, self.remote_directory)
            command = self._with_docker_machine(
                '%s inspect --format="%s" "%s"'
                % (self.docker_command, "{{json .HostConfig.Binds}}", self.instance_name)
            )
            output = self._run_command(command).strip()
            binds = json.loads(output)
            is_running = any(dir_string in b for b in binds)

        # If we have a network, make sure the running container is using the
        # correct network (if not we need to create a new container on the
        # correct network).
        # Note: with proper naming conventions this shouldn't happen much.
        if self.network and is_running:
            command = self._with_docker_machine(
                '%s inspect --format="%s" "%s"'
                % (self.docker_command, "{{.NetworkSettings.Networks}}", self.instance_name)
            )
            output = self._run_command(command).strip()
            is_running = self.network in output
        return is_running

    def _run_silent_command(self, command, ignore_output=False):
        if ignore_output:
            return subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        else:
            return subprocess.call(command, stdout=sys.stderr, shell=True)

    def _run_command(self, command):
        return subprocess.check_output(command, shell=True).decode()

    def _string_exists(self, string, output):
        regex = re.compile(r"\b(?=\w)%s\b(?!\w)" % re.escape(string))
        return any(regex.findall(output))

    def _image_exists(self):
        """Checks if the dazel image exists in the local repository."""
        image = "%s/%s" % (self.repository, self.image_name)
        command = self._with_docker_machine("%s image ls %s" % (self.docker_command, image))
        output = self._run_command(command)
        exists = len(output.splitlines()) > 1
        return exists

    def _get_build_command(self) -> str:
        """Returns the command to build the dazel image.

        Raises:
            RuntimeError: if the Dockerfile does not exist.
        """
        if not os.path.exists(self.dockerfile):
            raise RuntimeError("No Dockerfile specified to dazel image build command.")

        command = (
            "{docker_command} build {docker_build_args} "
            "--tag '{repository}/{image_name}:{image_tag}' "
            "--file '{dockerfile}' '{directory}'"
        ).format(
            docker_command=self.docker_command,
            docker_build_args=self.docker_build_args,
            repository=self.repository,
            image_name=self.image_name,
            image_tag=self.image_tag,
            dockerfile=self.dockerfile,
            directory=self.directory,
        )

        return self._with_docker_machine(command)

    def _build(self):
        """Builds the dazel image from the local dockerfile."""
        return self._run_silent_command(self._get_build_command())

    def _pull(self):
        """Pulls the relevant image from the dockerhub repository."""
        if not self.repository:
            raise RuntimeError("No repository to pull the dazel image from.")

        command = "%s pull %s/%s" % (self.docker_command, self.repository, self.image_name)
        command = self._with_docker_machine(command)
        return self._run_silent_command(command)

    def _is_predefined_network(self):
        """Checks if the network is one of the default existing docker network types"""
        return self.network in DOCKER_SPECIAL_NETWORK_NAMES

    def _network_exists(self):
        """Checks if the network we need to use exists."""
        command = self._with_docker_machine("%s network ls" % (self.docker_command))
        output = self._run_command(command)
        return self._string_exists(self.network, output)

    def _start_network(self):
        """Starts the docker network the container will use."""
        if not self.network:
            return 0

        command = "%s network create %s" % (self.docker_command, self.network)
        command = self._with_docker_machine(command)
        return self._run_silent_command(command)

    def _start_run_deps(self):
        """Starts the containers that are marked as runtime dependencies."""
        for run_dep_image, run_dep_name in self.run_deps:
            run_dep_instance = DockerInstance(
                instance_name=run_dep_name,
                image_name=run_dep_image,
                run_command=None,
                docker_command=None,
                dockerfile=None,
                repository=None,
                directory=None,
                command=None,
                volumes=None,
                ports=None,
                gpus=None,
                platform=None,
                shm_size=None,
                network=self.network,
                run_deps=None,
                docker_compose_file=None,
                docker_compose_command=None,
                docker_compose_project_name=None,
                docker_compose_services=None,
                bazel_rc_file=None,
                bazel_user_output_root=None,
                docker_run_privileged=self.docker_run_privileged,
                docker_machine=self.docker_machine,
                dazel_run_file=None,
            )
            if not run_dep_instance.is_running():
                logger.info("Starting run dependency: '%s' (name: '%s')" % (run_dep_image, run_dep_name))
                run_dep_instance._run_container()

    def _start_compose_services(self):
        """Starts the docker-compose services."""
        if not self.docker_compose_file:
            return 0

        command = "COMPOSE_PROJECT_NAME=%s %s -f %s pull --ignore-pull-failures %s" % (
            self.docker_compose_project_name,
            self.docker_compose_command,
            self.docker_compose_file,
            self.docker_compose_services,
        )
        command += " && COMPOSE_PROJECT_NAME=%s %s -f %s build %s" % (
            self.docker_compose_project_name,
            self.docker_compose_command,
            self.docker_compose_file,
            self.docker_compose_services,
        )
        command += " && COMPOSE_PROJECT_NAME=%s %s -f %s up --force-recreate -d %s" % (
            self.docker_compose_project_name,
            self.docker_compose_command,
            self.docker_compose_file,
            self.docker_compose_services,
        )
        command = self._with_docker_machine(command)
        return self._run_silent_command(command)

    def _run_container(self):
        """Runs the container itself."""
        logger.info("Starting docker container '%s'..." % self.instance_name)
        command = "%s stop %s" % (self.docker_command, self.instance_name)
        self._run_silent_command(self._with_docker_machine(command), ignore_output=True)
        command = "%s rm %s" % (self.docker_command, self.instance_name)
        self._run_silent_command(self._with_docker_machine(command), ignore_output=True)
        command = "%s run -id --name=%s %s %s %s %s %s %s %s %s %s %s %s%s %s" % (
            self.docker_command,
            self.instance_name,
            ("--platform=%s" % self.platform) if self.platform else "",
            "--privileged" if self.docker_run_privileged else "",
            ("--user=%s" % self.user if self.user else ""),
            ("-w %s" % self.remote_directory if self.remote_directory else ""),
            self.volumes,
            self.ports,
            self.env_vars,
            self.gpus,
            ("--shm-size=%s" % self.shm_size) if self.shm_size else "",
            ("--net=%s" % self.network) if self.network else "",
            ("%s/" % self.repository) if self.repository else "",
            ("%s:%s" % (self.image_name, self.image_tag)),
            self.run_command if self.run_command else "",
        )

        rc = self._run_silent_command(self._with_docker_machine(command))
        if rc:
            return rc

        # Touch the dazel run file to change the timestamp.
        if self.dazel_run_file:
            open(self.dazel_run_file, "w").write(self.instance_name + "\n")
            logger.info("Done.")

        return rc

    def _add_volumes(self, volumes):
        """Add the given volumes to the run string, and the bazel volumes we need anyway."""
        # This can only be intentional in code, so ignore None volumes.
        self.volumes = ""
        if volumes is None:
            return

        # DAZEL_VOLUMES can be a python iterable or a comma-separated string.
        if isinstance(volumes, str):
            volumes = [v.strip() for v in volumes.split(",")]
        elif volumes and not isinstance(volumes, collections.abc.Iterable):
            raise RuntimeError("DAZEL_VOLUMES must be comma-separated string or python iterable of strings")

        # Find the real source and output directories.
        real_directory = os.path.realpath(self.directory)
        volumes += [
            "%s:%s" % (real_directory, self.remote_directory),
        ]

        # If the user hasn't explicitly set a DAZEL_BAZEL_USER_OUTPUT_ROOT for
        # bazel, set it from the output directory so that we get the build
        # results on the host.
        real_bazelout = os.path.realpath(os.path.join(self.directory, "bazel-out", ".."))

        if not self.bazel_user_output_root and "/_bazel" in real_bazelout:
            parts = real_bazelout.split("/_bazel")
            first_part = parts[0]
            second_part = "/_bazel" + parts[1].split("/")[0]
            self.bazel_user_output_root = first_part + second_part

        # Add the bazel user output directory if it exists,
        # or the real bazelout directory if it does not
        if self.bazel_user_output_root:
            bazel_output_base = os.path.realpath(os.path.join(self.bazel_user_output_root, self.workspace_hex_digest))
            self.bazel_output_base = self._get_remote_directory(bazel_output_base)

            # Add the default bazel user output paths only if delegated volume is enabled
            if self.delegated_volume_flag:
                user_output_paths = DEFAULT_BAZEL_USER_OUTPUT_PATHS + [os.path.basename(real_directory)]
                for user_output_path in user_output_paths:
                    real_user_output_path = os.path.realpath(os.path.join(bazel_output_base, user_output_path))
                    if not os.path.isdir(real_user_output_path):
                        os.makedirs(real_user_output_path)
                    volumes += [
                        "%s:%s%s"
                        % (
                            real_user_output_path,
                            self._get_remote_directory(real_user_output_path),
                            self.delegated_volume_flag,
                        )
                    ]
        elif real_bazelout:
            volumes += ["%s:%s%s" % (real_bazelout, real_bazelout, self.delegated_volume_flag)]
            self.bazel_output_base = real_bazelout

        # Make sure the path exists on the host.
        if self.bazel_user_output_root and not os.path.isdir(self.bazel_user_output_root):
            os.makedirs(self.bazel_user_output_root)

        # Calculate the volumes string.
        self.volumes = '-v "%s"' % '" -v "'.join(volumes)

    def _get_remote_directory(self, local_directory, add_drive=False):
        remote_directory = local_directory
        if sys.platform == "win32":
            win_path = os.path.splitdrive(local_directory)[1]
            if add_drive:
                drive = os.path.splitdrive(local_directory)[0].strip(":")
                win_path = "/%s/%s" % (drive, win_path)
            remote_directory = str(pathlib.PureWindowsPath(win_path).as_posix())
        return remote_directory

    def _add_ports(self, ports):
        """Add the given ports to the run string."""
        # This can only be intentional in code, so disregard.
        self.ports = ""
        if not ports:
            return

        # DAZEL_PORTS can be a python iterable or a comma-separated string.
        if isinstance(ports, str):
            ports = [p.strip() for p in ports.split(",")]
        elif ports and not isinstance(ports, collections.abc.Iterable):
            raise RuntimeError("DAZEL_PORTS must be comma-separated string or python iterable of strings")

        # calculate the ports string
        self.ports = '-p "%s"' % '" -p "'.join(ports)

    def _add_gpus(self, gpus):
        """Add the given ports to the run string."""
        # This can only be intentional in code, so disregard.
        self.gpus = ""
        if not gpus:
            return

        # DAZEL_GPUS can be a python iterable or a comma-separated string.
        if isinstance(gpus, str):
            gpus = [g.strip() for g in gpus.split(",")]
        elif gpus and not isinstance(gpus, collections.abc.Iterable):
            raise RuntimeError("DAZEL_GPUS must be comma-separated string or python iterable of strings")

        # calculate the gpus string
        self.gpus = "--gpus %s" % ",".join(gpus)

    def _add_env_vars(self, env_vars):
        """Add the given env vars to the run string."""
        # This can only be intentional in code, so disregard.
        self.env_vars = ""
        if not env_vars:
            return

        # DAZEL_ENV_VARS can be a python iterable or a comma-separated string.
        if isinstance(env_vars, str):
            env_vars = [p.strip() for p in env_vars.split(",")]
        elif env_vars and not isinstance(env_vars, collections.abc.Iterable):
            raise RuntimeError("DAZEL_ENV_VARS must be comma-separated string or python iterable of strings")

        # calculate the env string
        self.env_vars = '-e "%s"' % '" -e "'.join(env_vars)

    def _add_run_deps(self, run_deps):
        """Adds the necessary runtime container dependencies to launch."""
        # This can only be intentional in code, so disregard.
        self.run_deps = ""
        if not run_deps:
            return

        # DAZEL_RUN_DEPS can be a python iterable or a comma-separated string.
        if isinstance(run_deps, str):
            run_deps = [rd.strip() for rd in run_deps.split(",")]
        elif run_deps and not isinstance(run_deps, collections.abc.Iterable):
            raise RuntimeError("DAZEL_RUN_DEPS must be comma-separated string or python iterable of strings")

        def extract_image_and_instance(run_dep):
            if "::" in run_dep:
                return tuple(run_dep.split("::"))
            return (run_dep, self.network + "_" + run_dep.replace("/", "_").replace(":", "_"))

        self.run_deps = [extract_image_and_instance(rd) for rd in run_deps]

    def _add_compose_services(self, docker_compose_services):
        """Add the given services to the docker-compose up string."""
        # This can only be intentional in code, so ignore None services.
        self.docker_compose_services = ""
        if not docker_compose_services:
            return

        # DAZEL_DOCKER_COMPOSE_SERVICES can be a python iterable or a
        # comma-separated string.
        if isinstance(docker_compose_services, str):
            docker_compose_services = [s.strip() for s in docker_compose_services.split(",")]
        elif docker_compose_services and not isinstance(docker_compose_services, collections.abc.Iterable):
            raise RuntimeError(
                "DAZEL_DOCKER_COMPOSE_SERVICES must be comma-separated string or python iterable of strings"
            )

        # Create the actual services string.
        self.docker_compose_services = " ".join(docker_compose_services)

    def _docker_exists(self):
        """Checks if the basic docker executable exists."""
        return self._command_exists(self.docker_command)

    def _docker_compose_exists(self):
        """Checks if the docker-compose executable exists."""
        return self._command_exists(self.docker_compose_command)

    def _command_exists(self, cmd):
        """Checks if a command exists on the system."""
        rc = shutil.which(cmd)
        return rc is not None

    def _with_docker_machine(self, cmd):
        if self.docker_machine is None or not self._command_exists("docker-machine"):
            return cmd
        return "eval $(docker-machine env %s) && (%s)" % (self.docker_machine, cmd)

    @classmethod
    def _config_from_files(cls) -> Dict[str, Any]:
        """Creates a configuration from a base .dazelrc and overriding user.dazelrc file (if exists)."""
        directory = cls.find_workspace_directory()
        local_dazelrc_path = os.path.join(directory, DAZEL_RC_FILE)
        dazelrc_path = os.environ.get("DAZEL_RC_FILE", local_dazelrc_path)

        if not os.path.exists(dazelrc_path):
            return {"DAZEL_DIRECTORY": os.environ.get("DAZEL_DIRECTORY", directory)}

        config = {}
        with open(dazelrc_path, "r") as dazelrc:
            exec(dazelrc.read(), config)
        config["DAZEL_DIRECTORY"] = os.environ.get("DAZEL_DIRECTORY", directory)

        user_dazelrc_path = os.path.join(directory, "user.dazelrc")
        if os.path.exists(user_dazelrc_path):
            with open(user_dazelrc_path, "r") as user_dazelrc:
                exec(user_dazelrc.read(), config)

        return config

    @classmethod
    def _config_from_environment(cls):
        """Creates a configuration from environment variables."""
        return {name: value for (name, value) in os.environ.items() if name.startswith("DAZEL_")}

    @classmethod
    def find_workspace_directory(cls):
        """Find the workspace directory.

        This is done by traversing the directory structure from the given dazel
        directory until we find the WORKSPACE file.
        """
        directory = os.path.realpath(os.environ.get("DAZEL_DIRECTORY", DEFAULT_DIRECTORY))
        root_dir = os.path.join(os.path.splitdrive(os.getcwd())[0], os.sep)
        while directory and directory != root_dir and not os.path.exists(os.path.join(directory, BAZEL_MODULE_FILE)):
            directory = os.path.dirname(directory)
        if not os.path.exists(os.path.join(directory, BAZEL_MODULE_FILE)):
            raise FileNotFoundError("ERROR: No %s file found!" % BAZEL_MODULE_FILE)
        else:
            return directory


def _update_apt_lockfiles(di: DockerInstance) -> int:
    """Regenerate apt package lockfiles to keep them in sync with upstream repositories.

    Scans MODULE.bazel for apt.install() declarations and regenerates each
    lockfile unconditionally. This ensures lockfiles stay current even when
    upstream package versions are removed or superseded (e.g., security patches
    replacing older .deb URLs).

    Args:
        di: DockerInstance object

    Returns:
        Exit code (0 on success)
    """
    workspace_dir = di.find_workspace_directory()
    module_bazel_path = os.path.join(workspace_dir, BAZEL_MODULE_FILE)

    if not os.path.exists(module_bazel_path):
        return 0

    with open(module_bazel_path, "r") as f:
        content = f.read()

    apt_install_pattern = re.compile(r"apt\.install\s*\((.*?)\)", re.DOTALL)
    name_pattern = re.compile(r'name\s*=\s*"([^"]+)"')

    for match in apt_install_pattern.finditer(content):
        block = match.group(1)

        name_match = name_pattern.search(block)
        if not name_match:
            continue

        repo_name = name_match.group(1)

        logger.info("Regenerating lockfile for @%s..." % repo_name)
        rc = di.send_command(["run", "@%s//:lock" % repo_name], quiet=True)
        if rc:
            logger.error("ERROR: Failed to regenerate lockfile for @%s with exit code %d" % (repo_name, rc))
            return rc
        logger.info("Lockfile for @%s regenerated successfully!" % repo_name)

    return 0


def _update_requirements_txt(di: DockerInstance) -> int:
    """Update requirements.txt if it's out of sync with requirements.in.

    Uses the :requirements.test target to check if requirements.txt needs updating.

    Args:
        di: DockerInstance object

    Returns:
        Exit code (0 on success or if no update needed)
    """
    # Test if requirements.txt is in sync - suppress output since we expect it may fail
    logger.info("Checking if requirements.txt is in sync...")
    test_result = di.send_command(["test", "//:requirements.test", "--test_output=errors"])

    if test_result == 0:
        # Already in sync, nothing to do
        return 0

    logger.info("requirements.txt is out of sync, updating...")

    rc = di.send_command(["run", "//:requirements.update", "--", "--upgrade"])
    if rc:
        logger.error("ERROR: Failed to update requirements.txt with exit code %d" % rc)
        return rc

    logger.info("requirements.txt updated successfully!")
    return 0


def _run_linter(di: DockerInstance, args: List[str], fix_mode: bool = False) -> int:
    """Run the linter script with the given arguments.

    Args:
        di: DockerInstance object
        args: Command arguments after lint/fix command (targets and flags)
        fix_mode: If True, runs with --apply-patches flag

    Returns:
        Exit code from linter script
    """
    targets_str = " ".join(args) if args else "default targets (//...)"
    logger.info("Running linter on targets: %s" % targets_str)

    workspace_dir = di.find_workspace_directory()
    linter_script = os.path.join(workspace_dir, "build/lint/linter.sh")

    cmd_args = [linter_script]
    if fix_mode:
        cmd_args.append("--apply-patches")
    cmd_args.extend(args)

    try:
        rc = subprocess.call(cmd_args, cwd=workspace_dir)
    except (subprocess.SubprocessError, OSError) as e:
        logger.error("ERROR: Failed to run linter script: %s", e)
        return 1

    if rc:
        logger.error("ERROR: Linter failed with exit code %d" % rc)

    return rc


def main():
    # Read the configuration either from .dazelrc or from the environment.
    di = DockerInstance.from_config()

    # If there is no .dazel_run file, or it is too old, start the DockerInstance.
    if (
        not os.path.exists(di.dazel_run_file)
        or not di.is_running()
        or (os.path.exists(di.dockerfile) and os.path.getctime(di.dockerfile) > os.path.getctime(di.dazel_run_file))
    ):
        rc = di.start()
        if rc:
            return rc

    # Check if this is a "lint" command
    if len(sys.argv) > 1 and sys.argv[1] == "lint":
        rc = _run_linter(di, sys.argv[2:])
        if rc == 0:
            logger.info("Lint command completed successfully!")
        return rc

    # Check if this is a "fix" command
    if len(sys.argv) > 1 and sys.argv[1] == "fix":
        # Update requirements.txt if out of sync with requirements.in
        rc = _update_requirements_txt(di)
        if rc:
            return rc

        # Regenerate apt package lockfiles to stay in sync with upstream repositories
        rc = _update_apt_lockfiles(di)
        if rc:
            return rc

        # Run the linter with fix mode to apply patches
        rc = _run_linter(di, sys.argv[2:], fix_mode=True)
        if rc:
            return rc

        # Run the formatter
        logger.info("Running formatter...")
        formatter_args = ["run", "//build/format:format_multirun"]
        rc = di.send_command(formatter_args)
        if rc:
            logger.error("ERROR: Formatter failed with exit code %d" % rc)
            return rc

        # Rerun the linter to check if there are any remaining violations
        rc = _run_linter(di, sys.argv[2:])
        if rc:
            logger.error("Fix command completed with remaining violations. Please fix them manually.")
            return rc

        logger.info("Fix command completed successfully! No remaining violations found.")
        return 0

    # Forward the command line arguments to the container.
    return di.send_command(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
