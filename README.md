# Image Processing Project

A collection of custom image processing functions built from scratch without relying on high-level image manipulation libraries.

## Overview

This project implements fundamental image processing operations including zooming, shrinking, and bit-depth quantization on grayscale images.

## Features

### 1. Image Zoom & Shrink

- **Zoom**: Enlarges images to 4x their original size using replication
- **Shrink**: Reduces images to 1/4 their original size using replication
- **Note**: Converts color images to grayscale; custom implementation without built-in zoom/shrink functions

### 2. Bit-Depth Quantization

- Reduces 8-bit grayscale images to:
  - 1-bit (black & white)
  - 3-bit
  - 6-bit
- **Note**: Custom quantization implementation without built-in functions

## Test Images

The following test images are used to validate functionality:

- `flowers.jpg`
- `cartoon.jpg`
- `kittens.jpg`
- `cherry_tree.jpg`

## Files

- `function_zoom.py` - Main implementation of image processing functions
- `images/` - Directory containing test images and output results

## Deliverables

All source code and result images are provided in a formatted PDF document: `studentID_assignment1.pdf`
