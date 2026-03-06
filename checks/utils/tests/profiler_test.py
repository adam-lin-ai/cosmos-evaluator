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

"""Unit tests for profiler module."""

import time
import unittest
from unittest.mock import MagicMock, patch

from checks.utils.profiler import Profiler


class TestProfiler(unittest.TestCase):
    """Test cases for Profiler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.profiler = Profiler()

    def test_init(self):
        """Test Profiler initialization."""
        self.assertEqual(len(self.profiler.timings), 0)
        self.assertIsNone(self.profiler.current_timer)
        self.assertIsNone(self.profiler.current_operation)

    def test_start_operation(self):
        """Test starting a timing operation."""
        operation_name = "test_operation"

        with patch("time.time", return_value=100.0):
            self.profiler.start(operation_name)

        self.assertEqual(self.profiler.current_operation, operation_name)
        self.assertEqual(self.profiler.current_timer, 100.0)

    def test_end_operation(self):
        """Test ending a timing operation."""
        operation_name = "test_operation"

        with patch("time.time", side_effect=[100.0, 105.0]):  # start=100, end=105
            self.profiler.start(operation_name)
            self.profiler.end()

        # Operation should be recorded
        self.assertIn(operation_name, self.profiler.timings)
        self.assertEqual(len(self.profiler.timings[operation_name]), 1)
        self.assertEqual(self.profiler.timings[operation_name][0], 5.0)

        # Current operation should be cleared
        self.assertIsNone(self.profiler.current_timer)
        self.assertIsNone(self.profiler.current_operation)

    def test_multiple_operations_same_name(self):
        """Test timing multiple operations with the same name."""
        operation_name = "repeated_operation"

        # First operation: 100 -> 103 (3 seconds)
        with patch("time.time", side_effect=[100.0, 103.0]):
            self.profiler.start(operation_name)
            self.profiler.end()

        # Second operation: 200 -> 207 (7 seconds)
        with patch("time.time", side_effect=[200.0, 207.0]):
            self.profiler.start(operation_name)
            self.profiler.end()

        # Should have two timing records
        self.assertEqual(len(self.profiler.timings[operation_name]), 2)
        self.assertEqual(self.profiler.timings[operation_name][0], 3.0)
        self.assertEqual(self.profiler.timings[operation_name][1], 7.0)

    def test_start_overlapping_operations(self):
        """Test starting a new operation before ending the current one."""
        # Start first operation
        with patch("time.time", side_effect=[100.0, 105.0, 110.0]):
            self.profiler.start("operation1")
            # Start second operation - should end first automatically
            self.profiler.start("operation2")

        # First operation should be recorded (ended automatically)
        self.assertIn("operation1", self.profiler.timings)
        self.assertEqual(len(self.profiler.timings["operation1"]), 1)
        self.assertEqual(self.profiler.timings["operation1"][0], 5.0)

        # Current operation should be operation2
        self.assertEqual(self.profiler.current_operation, "operation2")
        self.assertEqual(self.profiler.current_timer, 110.0)

    def test_end_without_start(self):
        """Test ending operation without starting one."""
        # Should not raise an exception
        self.profiler.end()

        # Should not create any timing records
        self.assertEqual(len(self.profiler.timings), 0)

    def test_get_summary_empty(self):
        """Test getting summary with no operations."""
        summary = self.profiler.get_summary()
        self.assertEqual(summary, {})

    def test_get_summary_single_operation(self):
        """Test getting summary with single operation."""
        operation_name = "test_op"

        with patch("time.time", side_effect=[100.0, 105.0]):
            self.profiler.start(operation_name)
            self.profiler.end()

        summary = self.profiler.get_summary()

        self.assertIn(operation_name, summary)
        op_stats = summary[operation_name]
        self.assertEqual(op_stats["total_time"], 5.0)
        self.assertEqual(op_stats["avg_time"], 5.0)
        self.assertEqual(op_stats["min_time"], 5.0)
        self.assertEqual(op_stats["max_time"], 5.0)
        self.assertEqual(op_stats["count"], 1)

    def test_get_summary_multiple_operations(self):
        """Test getting summary with multiple operations."""
        # Operation 1: 3 calls with times [2.0, 4.0, 6.0]
        with patch("time.time", side_effect=[100.0, 102.0]):  # 2.0 seconds
            self.profiler.start("op1")
            self.profiler.end()

        with patch("time.time", side_effect=[200.0, 204.0]):  # 4.0 seconds
            self.profiler.start("op1")
            self.profiler.end()

        with patch("time.time", side_effect=[300.0, 306.0]):  # 6.0 seconds
            self.profiler.start("op1")
            self.profiler.end()

        # Operation 2: 1 call with time [10.0]
        with patch("time.time", side_effect=[400.0, 410.0]):  # 10.0 seconds
            self.profiler.start("op2")
            self.profiler.end()

        summary = self.profiler.get_summary()

        # Check operation 1 stats
        op1_stats = summary["op1"]
        self.assertEqual(op1_stats["total_time"], 12.0)  # 2+4+6
        self.assertEqual(op1_stats["avg_time"], 4.0)  # 12/3
        self.assertEqual(op1_stats["min_time"], 2.0)
        self.assertEqual(op1_stats["max_time"], 6.0)
        self.assertEqual(op1_stats["count"], 3)

        # Check operation 2 stats
        op2_stats = summary["op2"]
        self.assertEqual(op2_stats["total_time"], 10.0)
        self.assertEqual(op2_stats["avg_time"], 10.0)
        self.assertEqual(op2_stats["min_time"], 10.0)
        self.assertEqual(op2_stats["max_time"], 10.0)
        self.assertEqual(op2_stats["count"], 1)

    def test_print_summary_empty(self):
        """Test printing summary with no operations."""
        mock_logger = MagicMock()

        self.profiler.print_summary(mock_logger)

        # Should not log anything for empty profiler
        mock_logger.info.assert_not_called()

    def test_print_summary_with_operations(self):
        """Test printing summary with operations."""
        # Add some operations
        with patch("time.time", side_effect=[100.0, 105.0]):  # 5.0 seconds
            self.profiler.start("slow_op")
            self.profiler.end()

        with patch("time.time", side_effect=[200.0, 201.0]):  # 1.0 second
            self.profiler.start("fast_op")
            self.profiler.end()

        mock_logger = MagicMock()

        self.profiler.print_summary(mock_logger)

        # Should have logged header, operations, and total
        self.assertTrue(mock_logger.info.called)

        # Check that log calls were made
        log_calls = mock_logger.info.call_args_list
        log_messages = [call[0][0] for call in log_calls]

        # Should include header
        header_found = any("PERFORMANCE PROFILING SUMMARY" in msg for msg in log_messages)
        self.assertTrue(header_found)

        # Should include operation names
        slow_op_found = any("slow_op" in msg for msg in log_messages)
        fast_op_found = any("fast_op" in msg for msg in log_messages)
        self.assertTrue(slow_op_found)
        self.assertTrue(fast_op_found)

        # Should include total time
        total_found = any("Total processing time" in msg for msg in log_messages)
        self.assertTrue(total_found)

    def test_print_summary_top_5_operations(self):
        """Test that print_summary only shows top 5 operations."""
        # Create 7 operations with different total times
        operations = [("op1", 10.0), ("op2", 9.0), ("op3", 8.0), ("op4", 7.0), ("op5", 6.0), ("op6", 5.0), ("op7", 4.0)]

        for op_name, duration in operations:
            with patch("time.time", side_effect=[100.0, 100.0 + duration]):
                self.profiler.start(op_name)
                self.profiler.end()

        mock_logger = MagicMock()

        self.profiler.print_summary(mock_logger)

        # Get all log messages
        log_calls = mock_logger.info.call_args_list
        log_messages = [call[0][0] for call in log_calls]
        all_logs = "\n".join(log_messages)

        # Top 5 operations should be mentioned
        for i in range(1, 6):  # op1 through op5
            self.assertIn(f"op{i}", all_logs)

        # Bottom 2 operations should not be mentioned in operation details
        # (they might appear in total calculation, but not as individual operations)
        op6_count = sum(1 for msg in log_messages if "op6:" in msg)
        op7_count = sum(1 for msg in log_messages if "op7:" in msg)
        self.assertEqual(op6_count, 0)
        self.assertEqual(op7_count, 0)

    def test_timing_precision(self):
        """Test that timing precision is maintained."""
        operation_name = "precision_test"

        # Use very small time difference
        with patch("time.time", side_effect=[100.0, 100.001]):  # 1ms
            self.profiler.start(operation_name)
            self.profiler.end()

        summary = self.profiler.get_summary()
        op_stats = summary[operation_name]

        # Should maintain precision
        self.assertAlmostEqual(op_stats["total_time"], 0.001, places=6)
        self.assertAlmostEqual(op_stats["avg_time"], 0.001, places=6)

    def test_real_timing_integration(self):
        """Integration test with real timing (not mocked)."""
        operation_name = "real_timing"

        self.profiler.start(operation_name)
        time.sleep(0.01)  # Sleep for 10ms
        self.profiler.end()

        summary = self.profiler.get_summary()
        op_stats = summary[operation_name]

        # Should be approximately 0.01 seconds (with some tolerance)
        self.assertGreater(op_stats["total_time"], 0.005)  # At least 5ms
        self.assertLess(op_stats["total_time"], 0.05)  # Less than 50ms


if __name__ == "__main__":
    unittest.main()
