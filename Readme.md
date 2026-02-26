# CMS-1500 Claim Form Extraction System

This project is a Flask-based web application designed to automatically extract structured information from CMS-1500 medical claim forms provided as PDF or image files. The system integrates Optical Character Recognition (OCR), computer vision techniques, and Natural Language Processing (NLP) models to convert unstructured claim documents into machine-readable JSON outputs.

---

## Project Overview

Manual processing of medical claim forms is time-consuming and prone to errors. This application automates the extraction of critical claim details such as patient information, insurance identifiers, diagnosis codes, procedure codes, service dates, and total charges. The solution leverages Tesseract OCR for text recognition, OpenCV for image preprocessing, and a HuggingFace transformer model for contextual understanding.

---

## Key Features

- Upload and process CMS-1500 claim forms in PDF or image format
- Automatic conversion of PDF pages into high-resolution images
- Coordinate-based extraction of predefined form fields
- OCR-based text recognition and preprocessing
- Regex-based cleaning and normalization of extracted data
- Recovery of diagnosis codes from full-document OCR text
- Output stored as structured JSON files
- Simple and intuitive Flask web interface

---

## Technology Stack

### Programming Language
- Python 3.9 or higher

### Backend Framework
- Flask

### OCR and Image Processing
- Tesseract OCR
- OpenCV
- pdf2image
- Pillow
- NumPy

### Natural Language Processing
- HuggingFace Transformers
- DistilBERT (fine-tuned on SQuAD)
- PyTorch

---

## Project Structure
