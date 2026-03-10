# The Story of Wages: A Data-Driven Journey

*Analyzing 4,165 workers to understand what really drives pay*

---

## Introduction: Who Are We Studying?

Imagine walking into a room filled with 4,165 workers from all walks of life. Some are just starting their careers, while others are nearing retirement. Some have PhDs, others didn't finish high school. Some work 52 weeks a year, others just a few months.

This is our dataset—a snapshot of workers from the US labor market. We have information about their experience, education, weeks worked, and importantly, their wages (logged, which makes statistical analysis easier). We also know details like their gender, whether they belong to a union, if they live in the South, and more.

Let's dive in and discover what this data tells us about the American worker.

---

## 1. Three Things the Data Reveals About Workers

### The Education Effect

Here's something that won't surprise anyone: **more education means more money**. The correlation between years of education and wages is 0.394—a solid moderate relationship. On average, each extra year of schooling adds about 7.3% to a worker's log wage. That may not sound like much, but over a full education journey, it adds up significantly. Someone with a college degree (16 years) earns substantially more than someone who dropped out at 12 years.

### Experience Matters—But There's a Catch

Work experience also pushes wages up, with a correlation of 0.209. But here's the interesting part: this relationship plateaus around 25 years. After that, more experience doesn't necessarily mean higher pay. Perhaps this reflects a point where seniority matters more than raw experience, or where older workers start to face different market dynamics.

### The Uncomfortable Truth About Gender

This one stings a bit. Looking at our data, men earn on average significantly more than women. The average log wage for men is around 6.73, compared to just 6.26 for women. This is a systemic issue that the data clearly exposes, regardless of how we feel about it.

---

## 2. What Does a Typical Worker's Wage Look Like?

Let's zoom in on our target variable: log wage (lwage). What does the distribution look like?

If we plot it out, we see something almost beautiful—a bell curve that's slightly off-center. The mean and median are nearly identical, suggesting a fairly symmetric distribution. 

But here's the twist: when we run a formal normality test (Shapiro-Wilk) on a sample, it gives us a p-value of **0.0028**. In the world of statistics, anything below 0.05 means we reject the idea of perfect normality. It's close enough for many statistical techniques, but not perfect—which is actually quite common in real-world data.

---

## 3. Cleaning Up the Messy Parts: Handling Outliers

Every dataset has its troublemakers—those unusual observations that don't fit the pattern. Let's find them.

Using the Interquartile Range (IQR) method, we identified **95 workers** (2.3% of our data) as outliers. Their wages fell either below 5.56 or above 7.79 in log terms. That's unusual—either they're earning very little or significantly more than typical.

In our final modeling, we decided to keep these outliers to ensure we capture the full spectrum of worker experiences, but it's a configurable choice. Outlier detection is often more art than science, and knowing they exist is the first step to understanding the extremes of the labor market.

---

## 4. Putting a Hypothesis to the Test

Let me share something I wondered when I first looked at this data: **Does joining a union actually pay off?**

It's a fair question. Unions negotiate for better wages, but do they actually deliver? Let's look at the numbers from our t-test:

- Union workers average log wage: **6.6817**
- Non-union workers average log wage: **6.6733**
- Difference: just **0.0084**

That difference is tiny. And the p-value came out to **0.527**—nowhere near statistical significance. 

So what does this tell us? In this particular dataset, union membership doesn't appear to translate to significantly higher wages on its own. It's a surprising finding that challenges common assumptions. Perhaps the benefits of unions lie elsewhere—in job security, better working conditions, or healthcare—factors our simple wage variable doesn't capture.

---

## 5. Building a Model to Predict Wages

Now for the fun part: can we build a machine that predicts what someone will earn?

I tried two approaches. First, a simple Linear Regression. It gave us an R² of 0.41, meaning it explains about 41% of the variation in wages. 

Then I tried a **Random Forest**. This model was much more sophisticated, achieving an R² of **0.62** and a Root Mean Squared Error (RMSE) of **0.2955**. It explains 62% of why people earn what they do.

What drives these predictions? Here's the "DNA" of a worker's wage according to our model:

1.  **Education** (The strongest predictor)
2.  **Experience** (A very close second)
3.  **Blue-collar Status** (Whether the job is manual or service-based)
4.  **Gender** (Being male still carries a predictive advantage)
5.  **Location** (Whether they live in a major metropolitan area - SMSA)

One concerning finding: our model shows that being Black is associated with lower predicted wages, even after controlling for education and experience. This points to systemic wage disparity in the data—a finding that warrants serious attention.

---

## 6. Finding Natural Groups: Who Are Your Colleagues?

One creative approach in data science is clustering—letting the data tell us what natural groups exist. We divided our 4,165 workers into 3 groups based on their experience, education, and how many weeks they work.

### Cluster 0: The Part-Timers (444 workers)
These workers have solid education (13.8 years) and mid-career experience (16.7 years), but they only work about **35 weeks a year**. They represent the part-time or seasonal workforce.

### Cluster 1: The Seasoned Veterans (1,497 workers)
These are our most experienced workers—**31 years** on average. They have slightly less formal education (11.4 years) but work nearly year-round (47.6 weeks). Their long tenure is their primary asset.

### Cluster 2: The High-Activity Professionals (2,224 workers)
Our largest group. These are younger to mid-career workers (12.9 years experience) with high education (13.7 years) who work almost every week of the year (**48.7 weeks**). They are the high-intensity core of the dataset.

This segmentation shows that there isn't one path to a good wage. Some climb the seniority ladder (Cluster 1); others leverage education and high activity (Cluster 2).

---

## 7 & 8. If We Needed to Sample: How Big Should Our Sample Be?

Suppose we didn't have access to all 4,165 workers. What if we only had the budget to survey a subset? 

Using Cochran's formula, we determined we'd need **352 respondents** to make valid inferences about the full population (95% confidence, 5% margin of error).

But we shouldn't just pick 352 people at random. Since our data is 88.7% male and 11.3% female, a simple random sample might accidentally undersample women. Instead, we should use **proportional stratified sampling**—surveying exactly 313 men and 40 women. This ensures our sample is a perfect "mini-me" of the whole population.

---

## 9. Why Do We Sample Instead of Studying Everyone?

**Sampling** is about being practical. We can't interview every worker in the country, but we can interview a representative few.

### The Upside:
- **Cost & Speed**: It's much cheaper and faster.
- **Quality**: With fewer respondents, we can spend more time ensuring their answers are accurate.
- **Feasibility**: Sometimes you can't reach everyone even if you wanted to.

### The Downside:
- **Sampling Error**: There's always a small risk the sample isn't perfectly representative.
- **Bias**: If you don't pick your sample carefully (like skipping rural workers), your results will be wrong.

---

## 10. What Makes or Breaks a Good Sample?

A good study isn't just about the numbers; it's about the design.

1.  **Sample Size**: Too small is unreliable; too big is a waste of money.
2.  **Sampling Method**: Stratified sampling is often better than random for representing minorities.
3.  **The Sampling Frame**: You need a good list to start with.
4.  **Variability**: If everyone is the same, you only need a tiny sample. If everyone is different, you need a big one.
5.  **Non-Response**: If half the people you ask say "no," you need to wonder if those who said "no" are different from those who said "yes."

---

## Closing Thoughts

This journey through the wages data taught us that **wages follow a story**. It's a story written in years of schooling, decades of experience, and the intensity of a work year. But it's also a story that still contains chapters of inequality based on gender and race.

Data science gives us the tools to read these stories. What we do with that knowledge—to build fairer workplaces and better careers—that's the next chapter.

---

## Files Created

| File | Description |
|------|-------------|
| `analysis.py` | Complete Python code (all analyses) |
| `wages_analysis_cleaned.csv` | Dataset with cleaned data |
| `REPORT.md` | Narrative report |
| `figures/q1_summary.png` | Education, experience, and gender charts |
| `figures/q2_distribution.png` | Wage distribution and Q-Q plot |
| `figures/q3_outliers.png` | Outlier detection visualization |
| `figures/q5_lr_actual_pred.png` | Linear Regression prediction accuracy |
| `figures/q5_rf_actual_pred.png` | Random Forest prediction accuracy |
| `figures/q6_clusters.png` | Visualization of the 3 worker segments |
