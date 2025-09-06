# Movie Recommendation System (Collaborative Filtering with Surprise)

A machine learning project that predicts **user–movie ratings** and can be extended to make **personalized movie recommendations**.  
This notebook is implemented with the **Surprise (scikit-surprise)** library using the **SVD** algorithm on the **MovieLens 100k** dataset.

---

## Tools & Libraries Used (from the notebook)
- **Python 3.x** & **Jupyter Notebook**
- **pandas** — data handling
- **numpy==1.26.4** — numerical ops (version pinned for Surprise compatibility)
- **surprise (scikit-surprise)** — recommendation toolkit
  - `Dataset` (with `load_builtin('ml-100k')`)
  - `SVD` (matrix factorization algorithm)
  - `model_selection.train_test_split` — split into train/test
  - `model_selection.cross_validate` — evaluate via CV
  - `model_selection.GridSearchCV` — hyperparameter tuning
  - `accuracy` — compute metrics like **RMSE** and **MAE**

> The notebook installs the packages with `pip install surprise`, `pip install numpy==1.26.4`, and `pip install pandas scikit-surprise`.

---

## Data
- Uses Surprise’s **built‑in** dataset: **MovieLens 100k** (`ml-100k`).  
- No external CSVs are required. Surprise downloads and caches the dataset automatically.

---

## Getting Started

### 1) Clone (optional)
```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2) Create & activate a virtual environment (recommended)
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3) Install dependencies
If you have a `requirements.txt` use:
```bash
pip install -r requirements.txt
```

Or install exactly what the notebook uses:
```bash
pip install numpy==1.26.4 pandas scikit-surprise jupyter
```

### 4) Open the notebook
```bash
jupyter notebook Copy_of_Movie_Recommendation_System.ipynb
```

---

## 🛠️ How the Model Works (as in the notebook)
1. **Load data**
   ```python
   from surprise import Dataset
   data = Dataset.load_builtin('ml-100k')
   df = pd.DataFrame(data.raw_ratings, columns=['user', 'item', 'rating', 'timestamp'])
   ```
2. **Train/test split & model training (SVD)**
   ```python
   from surprise import SVD
   from surprise.model_selection import train_test_split

   trainset, testset = train_test_split(data, test_size=0.2)
   algo = SVD()
   algo.fit(trainset)
   predictions = algo.test(testset)
   ```
3. **Evaluation**
   ```python
   from surprise import accuracy
   rmse = accuracy.rmse(predictions, verbose=True)
   mae  = accuracy.mae(predictions, verbose=True)
   ```
4. **Hyperparameter tuning (Grid Search)**
   ```python
   from surprise.model_selection import GridSearchCV

   param_grid = {
       'n_factors': [50, 100],
       'n_epochs': [20, 30],
       'lr_all': [0.005, 0.01],
       'reg_all': [0.02, 0.1]
   }
   gs = GridSearchCV(SVD, param_grid, measures=['RMSE', 'MAE'], cv=3)
   gs.fit(data)
   print(gs.best_score['RMSE'], gs.best_params['RMSE'])
   ```
5. **Predict a specific user–item rating**
   ```python
   user_id = '196'  # example
   item_id = '302'  # example
   pred = algo.predict(user_id, item_id)
   print(round(pred.est, 2))
   ```

> The notebook also prints the first few predictions and rounds `prediction.est` with Python’s built‑in `round()`.

---

## Example Results
- Metrics reported via Surprise’s `accuracy` (e.g., **RMSE**, **MAE**)
- Individual predictions printed for user–item pairs
- Ready to extend for **Top‑N recommendations** per user

---

## Future Improvements
- Generate **Top‑N** recommendations using the trained SVD model
- Add **model persistence** (save/load with `pickle`)
- Try other algorithms (`NMF`, `KNNBaseline`, etc. in Surprise)
- Add a **web UI** (Streamlit/Flask) to serve recommendations
- Switch to larger datasets (MovieLens 1M/20M) for better quality

---

## License
MIT (or your preferred license)

---

## Contributing
Issues and PRs are welcome!

