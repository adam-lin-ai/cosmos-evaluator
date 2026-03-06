# VLM Config Directory

This directory contains configuration files for the Vision Language Model (VLM) evaluation system.

## Configuration Files

| File | Description |
|------|-------------|
| `endpoints.json` | Configuration file containing API endpoint definitions for various Vision Language Models. Each endpoint includes base URL, model name, API key environment variable, and timeout settings. |
| `prompts.json` | Configuration file containing system and user prompts for VLM evaluation workflows. Includes preset checks for environment evaluation (weather, time of day, region, road conditions) and scenario checks for multi-stage evaluation pipeline (BEV-to-SFT conversion, QA generation, video QA, and scoring). |
