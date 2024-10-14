## Questions to ask

1. Correlation between posting datetime (weekdays/weekends) and engagement (ratio to followers/days since posted) received
2. Whether a link in a post/reel would impact the engagement/# of views?
3. Whether the type of content (posts vs. reels) impacts user engagement in relation to the number of followers/days since posted?
4. Correlation between the length of the video (reels) and the number of views


## Brief Summary (for discussion forum)

In this project, Ohm Avihingsanon and I would like to investigate the relationship between posting behaviors and engagement velocity on Instagram. Specifically, we are interested in how quickly early engagement (for now, it is vaguely defined as the number of likes and comments, normalized by followers, received within the first 7 days of posting) is received based on posting time (i.e., the day of a week, or weekdays vs. weekend, and the time in a day) and the presence of external links in the content. A sample conclusion might be be: "contents posted on Friday mornings typically gain engagement faster", or "including one or more external links directly in a post's content leads to slower initial engagement".

To achieve this objective, we would like to use a dataset of 2753 Instagram profiles belonging to some real estate agents in the lower mainland (for the sake of consistent timezone). The data was collected through a Python-based web scraping project (using Selenium) that I personally developed during my internship. While I am unable to share the scraper itself due to NDA constraints, I am permitted to scrape profiles and use the results at my own costs. If needed, we are open to explaining the scraping mechanism to assure the reliability of the dataset.

In the dataset, for each Instagram account, we have both metadata (e.g., Instagram ID, account type, counts of followers/following/posts biography, etc.), as well as content-specific data (e.g., post links, counts of likes/comments/views, posting datetime, caption, etc.) for up to 12 most recent _Posts_ (defined by Instagram as images, carousel of images, or _reels_) and up to 36 most recent _Reels_ (short-form videos). The dataset can be downloaded and preview [here](https://github.sfu.ca/mirrienl/CMPT353-Project/blob/3ad1e12da4f6215a56ea25ae2e937a66f6e37025/data/sample/sample.csv).

Please let us know if this topic is appropriate for our project, and if we can proceed to the next steps. Thank you!