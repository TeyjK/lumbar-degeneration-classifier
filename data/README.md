# Data

This project uses the RSNA 2024 Lumbar Spine Degenerative Classification dataset from Kaggle.

Dataset link: (add URL)

Due to size and licensing, the dataset is **not** stored in this repository.
To train or run inference:

1. Download the dataset from Kaggle.
2. Extract `train_images/` and `test_images/` into `data/`.
3. Ensure the structure matches:

   data/
     train_images/<study_id>/<series_id>/<instance>.dcm
     test_images/<study_id>/<series_id>/<instance>.dcm
