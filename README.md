# LightMHNN for Sensor-based Human Activity Recognition

## Project Structure

- **data/**: Contains the UCI-HAR dataset.
- **experiments/**: Contains training scripts.
- **models/**: Contains the MHNN model definition.
- **notebooks/**: (Optional) For interactive experiments.
- **test/**: Contains scripts for testing model performance.
- **utils/**: Contains utility scripts such as data loader and MDWD implementation.

## How to Run

1. Place the UCI-HAR dataset in the `data/` folder.

2. Train the model using:

   ```bash
   python experiments/train_mhnn.py
   ```

3. Test the model using:

   ```bash
   python test/data_analysis.py
   ```
