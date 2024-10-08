# Smart Travel Card Using Fingerprint

## Project Overview
This project presents an innovative biometric-based solution for storing and retrieving vaccination records securely on a smart card. Using a fingerprint authentication system, the project ensures that only authorized individuals can access their sensitive health information while traveling, without the risk of lost or damaged documents.

## Problem Statement
Travelers often need to present vaccination records when crossing borders. Traditional paper records can be easily lost, damaged, or tampered with, leading to inconvenience or health risks. There is a need for a secure, digital, and portable system that enables users to access their vaccination records anytime, anywhere.

## Solution
The **Smart Travel Card** integrates fingerprint-based biometric verification to ensure secure and convenient access to vaccination records. The smart card stores both the individual's biometric data (fingerprint minutiae) and their vaccination records. When the traveler scans their fingerprint, the system matches it against the stored biometric data, verifying their identity and granting access to the vaccination information.

### System Design

#### 1. Enrollment
In the enrollment phase, the system captures the user’s fingerprint data and processes it using a minutiae-based comparison technique. The extracted minutiae template is stored securely on the smart card, along with the individual’s vaccination records.

#### 2. Verification
During verification, the user presents the smart card (the unique ID) and scans their fingerprint. The system performs a one-to-one comparison between the scanned fingerprint and the pre-stored template. If match is confirmed, the user is granted access to their vaccination records.

#### 3. Vaccination Record Access
Once verified, the system retrieves and displays the user’s vaccination history from the card, ensuring convenience and security while traveling.

## Biometric Process

1. **Segmentation**: The system first separates the fingerprint image from the background noise.
2. **Ridge Orientation & Frequency**: It analyzes the ridges of the fingerprint for consistency and orientation.
3. **Enhancement**: The fingerprint image is enhanced to ensure clarity for minutiae extraction.
4. **Minutiae Detection**: Critical minutiae points (ridge endings and bifurcations) are detected.
5. **Matching**: The detected minutiae are compared with the stored template for verification.


## How It Works

1. **Step 1**: Enrollment – Capture and store the user’s fingerprint and vaccination data on the smart card.
2. **Step 2**: Verification – Authenticate the user using their fingerprint.
3. **Step 3**: Access Vaccination Records – Display the user’s vaccination information securely.

