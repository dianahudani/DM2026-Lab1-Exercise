## Report

### Data Inspection
I loaded the four-category newsgroup dataset (`comp.graphics`, `soc.religion.christian`, `sci.med`, `alt.atheism`) and ran `describe_data_tool` (`describe_data_5`). The box plot shows that document lengths are heavily right-skewed in every category, with a few extreme outliers (up to about 60,000 characters in `comp.graphics`).

### Document-Term Matrix and Filtering
I built the DTM with `build_dtm_tool` [state settings] and viewed a slice with `dtm_heatmap_tool` (`dtm_heatmap_11`). It is almost entirely zeros, so the matrix is very sparse.

I then ranked terms by variance with `variance_filter_tool` (`variance_filter_16`). The top terms are stopwords (`the`, `of`, `to`, `is`, `and`), plus noise tokens like `jpeg`.

In `feature_correlation_3`, the stopwords correlate strongly with each other (about 0.7 to 0.94), likely because they all scale with document length. `jpeg` and `edu` correlate weakly (about 0.1 to 0.45), probably because they come from headers and attachments.

### Dimensionality Reduction
I ran `reduce_dimensions_tool` (`reduce_dimensions_25`) with PCA [state parameters], colored by category. The categories overlap heavily in one dense cluster near the origin, with a few far-out points. The raw counts are dominated by stopword frequency and document length, not topic, so PCA does not separate the categories.

### Takeaway
Raw counts mostly reflect document length and common words. Stopword removal, TF-IDF or length normalization, noise-token removal, and a non-linear method like t-SNE or UMAP would likely help.