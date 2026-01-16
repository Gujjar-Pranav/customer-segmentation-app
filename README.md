# Customer Segmentation App (Production Grade)

This project is a production-grade customer segmentation pipeline built in Python using:
- Data Cleaning + Preprocessing
- Feature Engineering
- KMeans Clustering
- Model Evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Cluster Personas + Business Strategy Tables
- Cluster Profiling + Excel Reports
- Visualization (Cluster Counts, Heatmap, PCA 3D HTML)

---

## 📌 Project Structure

customer-segmentation-app/
config/
config.yaml
data/
marketing_campaign.xlsx
outputs/
(generated files will be saved here)
src/
main.py
io_utils.py
preprocessing.py
features.py
modeling.py
evaluation.py
reporting.py
categoricals.py
viz.py
pca_viz.py
validation.py
requirements.txt
README.md
Makefile
pyproject.toml

yaml
Copy code

---

## ✅ Requirements

Install dependencies inside your virtual environment:

```bash
pip install -r requirements.txt
✅ Input Dataset
Place your dataset file here:

bash
Copy code
data/marketing_campaign.xlsx
✅ Run Pipeline
Run using Python:

bash
Copy code
python -m src.main --config config/config.yaml
Run using Makefile (recommended):

bash
Copy code
make run
✅ Output Files (Generated Automatically)
All generated reports and files will be saved inside:

Copy code
outputs/
📌 Data Exports
File	Description
data_cleaned.xlsx	Cleaned raw data (missing + duplicates + caps applied)
data_featured.xlsx	Cleaned data + engineered features
data_clustered.xlsx	Featured data + cluster labels
data_clustered_named.xlsx	Cluster labels + cluster names
data_clustered_named_final.xlsx	Final full dataset with categorical grouping

📌 Evaluation Exports
File	Description
model_comparison.xlsx	Comparison of Hierarchical + GMM vs KMeans
pca_explained_variance.xlsx	PCA variance explained %
pca_loadings.xlsx	PCA loadings per feature

📌 Business Reporting Tables
File	Description
cluster_summary.xlsx	Main cluster summary table
cluster_personas.xlsx	Personas + traits + actions
revenue_contribution.xlsx	Revenue and customer contribution per cluster
rfm_summary.xlsx	Recency / Frequency / Monetary by cluster
promo_roi.xlsx	Promo response rate and deal dependency
channel_strategy.xlsx	Web/store channel preference by cluster
discount_risk.xlsx	Discount addiction risk by cluster
clv_summary.xlsx	CLV-lite proxy per cluster

📌 Categorical Distribution Tables
File	Description
cluster_marital_group_distribution.xlsx	Marital group distribution per cluster
cluster_education_group_distribution.xlsx	Education group distribution per cluster

📌 Plots / Visual Output
File	Description
cluster_counts.png	Cluster size bar chart
cluster_feature_heatmap.png	Normalized cluster feature heatmap
pca_3d.html	Interactive PCA 3D plot
pca_3d_centroids.html	Interactive PCA 3D plot with centroids

✅ Configuration
All project settings can be controlled from:

arduino
Copy code
config/config.yaml
Example settings:

clustering.k → number of clusters

clustering.scaler → StandardScaler / RobustScaler

run.save_tables → enable/disable Excel outputs

run.save_plots → enable/disable plots saving

run.show_plots → show plots or only save

✅ Code Quality Commands
Format code:

bash
Copy code
make format
Lint code:

bash
Copy code
make lint
Run full pipeline:

bash
Copy code
make run
✅ Notes
This pipeline is built for PyCharm production use

All results are reproducible using seed in config

The output folder will always contain the latest run results

👤 Author
Pranav Gujjar
Machine Learning Engineer