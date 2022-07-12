---
title: 'Watch your language - update: feature engineering'
date: '`r Sys.Date()`'
output:
  html_document:
    number_sections: true
    fig_caption: true
    toc: true
    fig_width: 7
    fig_height: 4.5
    theme: cosmo
    highlight: tango
    code_folding: hide
---

```{r setup, include=FALSE, echo=FALSE}
knitr::opts_chunk$set(echo=TRUE, error=FALSE)
```

**Update: Check out the new section on feature engineering.**


# Introduction

This is an extensive Exploratory Data Analysis for the [Google Text Normalization Challenge - English Language](https://www.kaggle.com/c/text-normalization-challenge-english-language) competition with [tidy R](http://tidyverse.org/), [ggplot2](http://ggplot2.tidyverse.org/), and [tidytext](https://cran.r-project.org/web/packages/tidytext/index.html).

The aim of this challenge is to "automate the process of developing text normalization grammars via machine learning" (see the challenge [description](https://www.kaggle.com/c/text-normalization-challenge-english-language)). [Text normalisation](https://en.wikipedia.org/wiki/Text_normalization) describes the process of transforming language into a specific, self-consistent grammar system with well-defined rules. Here, we aim to convert "written expressions into into appropriate 'spoken' forms", as described on the description page which provides the following example: "[...] convert 12:47 to 'twelve forty-seven' and $3.16 into 'three dollars, sixteen cents.'" 

The [data](https://www.kaggle.com/c/text-normalization-challenge-english-language/data) comes in the shape of two files: `../input/en_train.csv` and `../input/en_test.csv`. Each row contains a single language element (such as words, letters, or punctuation) together with its associated identifiers. The [evaluation metric](https://www.kaggle.com/c/text-normalization-challenge-english-language#evaluation) is the total percentage of correctly translated tokens. The example provided is that "if the input is '145' and the predicted output is 'one forty five' but the correct output is 'one hundred forty five', this is counted as a single error."

Note: This kernel will only study the *English language data*. For the *Russian language* competition see [here](https://www.kaggle.com/c/text-normalization-challenge-russian-language).

## Load libraries and helper functions

```{r, message = FALSE}
# general visualisation
library('ggplot2') # visualisation
library('scales') # visualisation
library('grid') # visualisation
library('gridExtra') # visualisation
library('RColorBrewer') # visualisation
library('corrplot') # visualisation
library('ggforce') # visualisation
library('treemapify') # visualisation

# general data manipulation
library('dplyr') # data manipulation
library('readr') # input/output
library('data.table') # data manipulation
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation

# Text / NLP
library('tidytext') # text analysis
library('tm') # text analysis
library('SnowballC') # text analysis
library('topicmodels') # text analysis
library('wordcloud') # test visualisation

# Extra Vis
library('plotly')
```



```{r}
#We use the *multiplot* function, courtesy of [R Cookbooks](http://www.cookbook-r.com/Graphs/Multiple_graphs_on_one_page_(ggplot2)/) to create multi-panel plots.

# Define multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
```


## Load data

We use *data.table's* fread function to speed up reading in the data. The training data file is rather extensive, with close to 10 million rows and 300 MB uncompressed size:

```{r warning=FALSE, results=FALSE}
train <- as.tibble(fread('../input/en_train.csv'))
test <- as.tibble(fread('../input/en_test.csv'))
```


## File structure and content

We will have an overview of the data sets using the *summary* and *glimpse* tools. First the training data:

```{r}
summary(train)
```


```{r}
glimpse(train)
```

And then the testing data:

```{r}
summary(test)
```


```{r}
glimpse(test)
```

We find:

- *Sentence\_id* is an integer identifying each individual sentence. These are the total number of sentences in the train and test data:

```{r}
print(c(max(train$sentence_id), max(test$sentence_id)))
```

- *Token\_id*, in a similar way, is an integer counting the number of language elements within each sentence; e.g. words or punctuation. The maximum *token\_id*, i.e. sentence length, is 255 in the training data and 248 in the test data.

- *Before* is the original language element and *after* is its normalised text. **The aim of this challenge is to predict the 'after' column for the test data set.**

- *Class* is indicating the type of the language token. This feature, together with *after*, is "intentionally omitted from the test set" ([data description](https://www.kaggle.com/c/text-normalization-challenge-english-language/data)). We will have a closer look at these classes shortly.


We combine the train and test data sets for comparison treatment:

```{r}
combine <- bind_rows(train %>% mutate(set = "train"),test %>% mutate(set = "test")) %>%
  mutate(set = as.factor(set))
```


## Missing values

There are no missing values in our data set:

```{r}
sum(is.na(train))
sum(is.na(test))
```


## Reformating features

We decide to turn *Class* into a factor for exploration purposes:

```{r}
train <- train %>%
  mutate(class = factor(class))
```


# Example sentences and overview visualisations

## A first look at normalised sentences

Before diving into data summaries, let's begin by printing a few *before* and *after* sentences to get a feeling for the data. To make this task easier, we first define a short helper function to compare these sentences:

```{r}
before_vs_after <- function(sent_id){

  bf <- train %>%
    filter(sentence_id == sent_id) %>%
    .$before %>%
    str_c(collapse = " ")
  
  af <- train %>%
    filter(sentence_id == sent_id) %>%
    .$after %>%
    str_c(collapse = " ")
  
  print(str_c("Before:", bf, sep = " ", collapse = " "))
  print(str_c("After :", af, sep = " ", collapse = " "))
}
```

Those are a few example sentences:

```{r}
before_vs_after(11)
```

```{r}
before_vs_after(99)
```

```{r}
before_vs_after(1234)
```

We already notice a few things:

- The first two examples are very similar, but the normalisations manage to make the difference between "april tenth" and "tenth of april" depending on how the date is written.

- "2015" becomes "twenty fifteen" instead of "two thousand fifteen". This should be easy to transform, though.

- "April" becomes "april". Lower vs upper case should not be a problem, but it is noteworthy.

- Acronyms like "PMO" turn into their spoken version of "p m o".

## Summary visualisations

Now let's look at overview visualisations. First,  we examine the different token *classes* and their frequency. Here is a visual summary: 

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 1", out.width="100%"}
train %>%
  group_by(class) %>%
  count() %>%
  ungroup() %>%
  mutate(class = reorder(class, n)) %>%
  ggplot(aes(class,n, fill = class)) +
  geom_col() +
  scale_y_log10() +
  labs(y = "Frequency") +
  coord_flip() +
  theme(legend.position = "none")
```

Note the logarithmic x-axis.

We find:

- The "PLAIN" *class* is by far the most frequent, followed by "PUNCT" and "DATE". 

- In total there are 16 *classes*, with "TIME", "FRACTION", and "ADDRESS" having the lowest number of occurences (around/below 100 tokens each).


Next up, this is the histogram distribution of the sentence length for the training data:

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 2", out.width="100%"}
train %>%
  group_by(sentence_id) %>%
  summarise(sentence_len = max(token_id)) %>%
  ggplot(aes(sentence_len)) +
  geom_histogram(bins = 50, fill = "red") +
  scale_y_sqrt() +
  scale_x_log10() +
  labs(x = "Sentence length")
```

We find that our sentences are typically up to 15-20 tokens long, after which the frequency drops quickly. Very long sentences (> 100 tokens) exist but are relatively rare. Note again the logarithmic x-axis and square-root y-axis.

Below we compare the sentence length distributions for the training vs test data sets, this time with an overlapping density plot:

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 3", out.width="100%"}
combine %>%
  group_by(sentence_id, set) %>%
  summarise(sentence_len = max(token_id)) %>%
  ggplot(aes(sentence_len, fill = set)) +
  geom_density(bw = 0.1, alpha = 0.5) +
  scale_x_log10() +
  labs(x = "Sentence length")
```

We find that the training data contains more shorter sentences (< 10 tokens) and that there is a larger proportion of longer sentences in the test data set. Note the logarithmic x-axis.

Check out a possible explanation by [Richard Sproat](https://www.kaggle.com/rwsproat) in the comment section below on why there aren't as many short sentences in the test data. His feedback also provides some insight into the data preparation process.


Next, we will look at the *token\_ids* of each *class* in their sentences:

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 4", out.width="100%"}
train %>%
  ggplot(aes(reorder(class, token_id, FUN = median), token_id, col = class)) +
  geom_boxplot() +
  scale_y_log10() +
  theme(legend.position = "none", axis.text.x  = element_text(angle=45, hjust=1, vjust=0.9)) +
  labs(x = "Class", y = "Token ID")

```

We find:

- The "TELEPHONE" class appears predominantly at *token\_ids* of less than 10.

- Above *token\_ids* of 100 we find no occurences of the "ELECTRONICS", "ADDRESS", "FRACTION", "DIGIT", "ORDINAL", "TIME", "MONEY", and "MEASURE" *classes*. Of those, "FRACTION", "MONEY", and "MEASURE" barely appear above `token_id == 20`.

- The *classes* "DECIMAL", "MONEY", "PUNCT", and "MEASURE" are rarely found in the first token of a sentence.


We can take this analysis a step further by relating the *token\_id* to the length of the sentence. Thereby, we will see at which relative position in a sentence a certain *class* is more likely to occur:

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 5", out.width="100%"}
sen_len <- train %>%
  group_by(sentence_id) %>%
  summarise(sentence_len = max(token_id))

train %>%
  left_join(sen_len, by = "sentence_id") %>%
  mutate(token_rel = token_id/sentence_len) %>%
  ggplot(aes(reorder(class, token_rel, FUN = median), token_rel, col = class)) +
  geom_boxplot() +
  #scale_y_log10() +
  theme(legend.position = "none", axis.text.x  = element_text(angle=45, hjust=1, vjust=0.9)) +
  labs(x = "Class", y = "Relative token ID")

```

We find:

- As suggest above, "TELEPHONE" tokens are more likely to occur early in a sentence. A similar observation holds for "LETTERS" and "PLAIN".

- Unsurprisingly, the "PUNCT" *class* can be found more frequently towards the end of a sentence. Similarly, "MONEY" tokens occur relatively late.

- In general, there is a certain trend among the *classes*, with medians ranging from about 0.4 to 0.8. However, interquartile ranges are wide and there is a large amount of overlap between the different *classes*.


# The impact of the normalisation

Now we will include the effects of the text normalisation in our study by analysing the changes it introduced in the training data.

To begin, we define the new feature *transformed* to indicate those tokens that changed from *before* to *after*:

```{r}
train <- train %>%
  mutate(transformed = (before != after))
```


In total, only about 7% of tokens in the training data, or about 660k objects in total, were changed during the process of text normalisation:

```{r}
train %>%
  group_by(transformed) %>%
  count() %>%
  mutate(freq = n/nrow(train))
```

This explains the high baseline accuracies we can achieve even without any adjustment of the test data input. 


By comparing the fraction of tokens that changed from *before* to *after* the text normalisation we can visualise which *classes* are most affected by this process:

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 6", out.width="100%"}
train %>%
  ggplot(aes(class, fill = transformed)) +
  geom_bar(position = "fill") +
  theme(axis.text.x  = element_text(angle=45, hjust=1, vjust=0.9)) +
  labs(x = "Class", y = "Transformed fraction [%]")
```

We find:

- Quite a few "ELECTRONIC" and "LETTERS" tokens did not change in *after* vs *before*.

- A noteable fraction of "PLAIN" *class* text elements **did change**.

- The majority of "VERBATIM"-*class* tokens remained identical during the normalisation.


This is a breakdown of the number and fraction how many tokens per *class* remained identical in *before* vs *after*. Here we only list the *classes* that feature at least one unchanged token:

```{r}
train %>%
  group_by(class, transformed) %>%
  count() %>%
  spread(key = transformed, value = n) %>%
  mutate(`TRUE` = ifelse(is.na(`TRUE`),0,`TRUE`),
         `FALSE` = ifelse(is.na(`FALSE`),0,`FALSE`)) %>%
  mutate(frac = `FALSE`/(`TRUE`+`FALSE`)*100) %>%
  filter(frac>1e-5) %>%
  arrange(desc(frac)) %>%
  rename(unchanged = `FALSE`, changed = `TRUE`, unchanged_percentage = frac)

```


# Token classes and their text normalisation

In order to explore the meaning of these classes for our normalisation task we modify our helper function to include the class name. Here we also remove punctuation:

```{r}
before_vs_after_class <- function(sent_id){

  bf <- train %>%
    filter(sentence_id == sent_id & class != "PUNCT") %>%
    .$before %>%
    str_pad(30) %>%
    str_c(collapse = " ")
  
  af <- train %>%
    filter(sentence_id == sent_id & class != "PUNCT") %>%
    .$after %>%
    str_pad(30) %>%
    str_c(collapse = " ")
  
  
  cl <- train %>%
    filter(sentence_id == sent_id & class != "PUNCT") %>%
    .$class %>%
    str_pad(30) %>%
    str_c(collapse = " ")
  
  print(str_c("[Class]:", cl, sep = " ", collapse = " "))
  print(str_c("Before :", bf, sep = " ", collapse = " "))
  print(str_c("After  :", af, sep = " ", collapse = " "))
}
```

Using our example sentence from earlier, we get the following output structure, indicating the combination of a "PLAIN" and a "DATE" token:

```{r}
before_vs_after_class(11)
```

In the following we will explore the different tokens *classes* within categories of similarity and provide a few examples for each.


## "PLAIN" *class*: modifications

As we saw above, most "PLAIN" tokens remained unchanged. However, about 0.5% were transformed, which still amounts to about 36k tokens:

```{r}
train %>%
  filter(class == "PLAIN") %>%
  group_by(transformed) %>%
  count() %>%
  spread(key = transformed, value = n) %>%
  mutate(frac = `FALSE`/(`TRUE`+`FALSE`)*100) %>%
  filter(!is.na(frac)) %>%
  arrange(desc(frac)) %>%
  rename(unchanged = `FALSE`, changed = `TRUE`, unchanged_percentage = frac)
```

Those are a few transformed examples:

```{r}
set.seed(1234)
train %>%
  filter(transformed == TRUE & class == "PLAIN") %>%
  sample_n(10)
```

We see a few typical changes, such as "-" to "to" and the adjustment from British to American spelling.


## Punctuation class: "PUNCT"

It is worth to briefly cross-check the "PUNCT" class. Intuitively, punctuation should not be affected by text normalisation unless it is associated to other structures such as numbers or dates. Here are a few punctuation examples:

```{r}
set.seed(1234)
train %>%
  filter(class == "PUNCT") %>%
  sample_n(10)
```

As expected, those tokens are identical *before* and *after*:

```{r}
train %>%
  filter(class == "PUNCT") %>%
  mutate(test = (before == after)) %>%
  group_by(test) %>%
  count()
```


## Numerical *classes* {.tabset}

By far the most diverse *classes* (and normalisation treatment) involve numbers in one way or another. Those classes are the following: "DATE", CARDINAL", "MEASURE", "ORDINAL", "DECIMAL", "MONEY", DIGIT", "TELEPHONE", "TIME", "FRACTION", and "ADDRESS". Here we will look at a few examples for each of them.


### "DATE" {-}

We have already seen how dates can be transformed differently depending on their formatting. Those are some more examples:

```{r}
before_vs_after_class(8)
```

```{r}
before_vs_after_class(12)
```


### "CARDINAL" {-}

Not a big surprise here: those are cardinal numbers. Interestingly, this includes Roman numerals:

```{r}
set.seed(1234)
train %>%
  select(-token_id, -transformed) %>%
  filter(class == "CARDINAL") %>%
  sample_n(5)
```

```{r}
before_vs_after(471926)
```


### "MEASURE" {-}

Those are mainly percentages and physical measurements:

```{r}
set.seed(1234)
train %>%
  select(-token_id, -transformed) %>%
  filter(class == "MEASURE") %>%
  sample_n(5)
```


```{r}
before_vs_after(200476)
```

```{r}
before_vs_after_class(225531)
```


### "ORDINAL" {-}

Also ordinal numbers can include Roman numerals, as in the example of Queen Elizabeth I:

```{r}
set.seed(1234)
train %>%
  select(-token_id, -transformed) %>%
  filter(class == "ORDINAL") %>%
  sample_n(5)
```

```{r}
before_vs_after(478452)
```


### "DECIMAL" {-}

```{r}
set.seed(1234)
train %>%
  select(-token_id, -transformed) %>%
  filter(class == "DECIMAL") %>%
  sample_n(5)
```

```{r}
before_vs_after(443910)
```

Also, in this particular example the token in parentheses is a "MEASURE":

```{r}
before_vs_after_class(443910)
```


### "MONEY" {-}

Money tokens come with currency symbols like "$" or plain text names like "yuan":

```{r}
set.seed(1234)
train %>%
  select(-token_id, -transformed) %>%
  filter(class == "MONEY") %>%
  sample_n(5)
```

```{r}
before_vs_after(90430)
```


### "DIGIT" {-}

```{r}
set.seed(1234)
train %>%
  select(-token_id, -transformed) %>%
  filter(class == "DIGIT") %>%
  sample_n(5)
```


```{r}
before_vs_after(481444)
```

Interesting. This example looks like a citation of an article or book, which means that "nineteen ninety six" should be correct instead of "one nine nine six", since the number most likely refers to the year of publication.


### "TELEPHONE" {-}

```{r}
set.seed(1234)
train %>%
  filter(class == "TELEPHONE") %>%
  select(-token_id, -class, -transformed) %>%
  sample_n(5)
```

```{r}
before_vs_after(88957)
```

That doesn't look like a telephone number to me. Neither does this one:

```{r}
before_vs_after(476323)
```

Quite a few of these entries seem to be in fact ISBN numbers. Everything that is formatted like integer digits plus dashes appears to be identified as a "TELEPHONE" token. This could be a tricky category.


### "TIME" {-}

The "TIME" formatting and normalisation are comparatively simple:

```{r}
set.seed(1234)
train %>%
  select(-token_id, -transformed) %>%
  filter(class == "TIME") %>%
  sample_n(5)
```


```{r}
before_vs_after(471678)
```


### "FRACTION" {-}

Here it become tricky again:

```{r}
set.seed(1234)
train %>%
  select(-token_id, -transformed) %>%
  filter(class == "FRACTION") %>%
  sample_n(5)
```

Many of those are unlikely to be fractions and are therefore normalised in a rather clumsy way:

```{r}
before_vs_after(470330)
```

But since we only need to reproduce this normalisation approach it should actually make it easier for us; because the required approach appears to be rather homogeneous for any two integers separated by a forward slash.


### "ADDRESS" {-}

The "ADDRESS" *class* appears to assigned to alpha-numeric combinations:

```{r}
set.seed(1234)
train %>%
  select(-token_id, -transformed) %>%
  filter(class == "ADDRESS") %>%
  sample_n(5)
```

```{r}
before_vs_after(88342)
```

```{r}
before_vs_after(440683)
```


## Acronyms and Initials: "LETTERS" *class*

The tokens in the *class* "LETTERS" appear to be normalised to a form which spells them out one by one:

```{r}
set.seed(4321)
train %>%
  select(-token_id, -transformed) %>%
  filter(class == "LETTERS") %>%
  sample_n(5)
```

```{r}
before_vs_after(571356)
```

Since those text elements are typically completely in upper case it should be relatively simple to define a normalisation. Given the relatively high frequency of the "LETTERS" *class* I suggest to use such a simple transformation as a first, a little more advanced LB baseline.


## Special symbols: "VERBATIM" *class*

Here we have special symbols such as non-english characters. Interestingly, not all of those are copied verbatim from *before* to *after*:


```{r}
set.seed(1234)
train %>%
  select(-token_id, -transformed) %>%
  filter(class == "VERBATIM") %>%
  sample_n(5)
```

The exception are Greek letters or the ampersand "&". It might be useful to search for other exceptions and make use of simple transformations such as "&" to "and".


## Websites: "ELECTRONIC" *class*

This *class* includes websites that are normalised using single characters and a "dot":

```{r}
set.seed(4321)
train %>%
  filter(class == "ELECTRONIC") %>%
  select(-token_id, -class, -sentence_id, -transformed) %>%
  sample_n(5)
```

```{r}
before_vs_after(99083)
```

This is another format that should be relatively easy to learn and implement.


**In summary:** The different *classes* appear to follow different translation rules for the text normalisation. Even though the *class* feature is not present in the *test* data set it might be useful to train a model to identify the specific class for a token and then apply its normalisation rules.


## Most frequent transformations per *class* - wordclouds {.tabset}

Here we are using the *wordcloud package* to visualise those *tokens* for each class that occur most frequently among the terms affected by the text normalisation. We start with the overall frequency in the first tab and then sequentially plot the individual *classes*.

### All classes {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 7", out.width="100%"}
train %>%
  filter(transformed == TRUE) %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 100, rot.per=0, fixed.asp=FALSE))
```

### "PLAIN" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 8", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "PLAIN") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 30, scale=c(4,1), rot.per=0, fixed.asp=FALSE))
```

### "DATE" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 9", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "DATE") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 50, scale=c(4,0.5), rot.per=0, fixed.asp=FALSE))
```

### "CARDINAL" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 10", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "CARDINAL") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 40, scale=c(4,1), rot.per=0, fixed.asp=FALSE))
```


### "MEASURE" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 11", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "MEASURE") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 50, scale=c(4,0.5), rot.per=0, fixed.asp=FALSE))
```


### "ORDINAL" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 12", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "ORDINAL") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 50, scale=c(4,0.5), rot.per=0, fixed.asp=FALSE))
```


### "DECIMAL" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 13", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "DECIMAL") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 50, scale=c(4,0.5), rot.per=0, fixed.asp=FALSE))
```

### "MONEY" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 14", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "MONEY") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 50, scale=c(4,0.5), rot.per=0, fixed.asp=FALSE))
```

### "DIGIT" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 15", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "DIGIT") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 30, scale=c(4,1), rot.per=0, fixed.asp=FALSE))
```

### "TELEPHONE" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 16", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "TELEPHONE") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 50, scale=c(4,0.5), rot.per=0, fixed.asp=FALSE))
```

### "TIME" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 17", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "TIME") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 50, scale=c(4,0.5), rot.per=0, fixed.asp=FALSE))
```

### "FRACTION" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 18", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "FRACTION") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 50, scale=c(4,0.5), rot.per=0, fixed.asp=FALSE))
```

### "ADDRESS" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 19", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "ADDRESS") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 50, scale=c(4,0.5), rot.per=0, fixed.asp=FALSE))
```

### "LETTERS" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 20", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "LETTERS") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 30, scale=c(4,1), rot.per=0, fixed.asp=FALSE))
```

### "VERBATIM" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 21", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "VERBATIM") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 30, scale=c(4,1), rot.per=0, fixed.asp=FALSE))
```

### "ELECTRONIC" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 22", out.width="100%"}
train %>%
  filter(transformed == TRUE, class == "ELECTRONIC") %>%
  count(before) %>%
  with(wordcloud(before, n, max.words = 30, scale=c(4,0.5), rot.per=0, fixed.asp=FALSE))
```


# Context matters: a next-neighbour analysis

In this section we turn to analysing whether the context of our normalised token can provide any indications towards its *class*. We're doing this by preparing a data frame in which every token is listed together with the token (and its *class*) that came immediately previous or next in the text data. (Note, that for the first/last tokens of a sentence the *previous/next* tokens will belong to the the previous/next sentence.)

We build this new data frame using the `dplyr` tool `lead` which shifts the contents of a vector by a certain number of indices (see also its cousin `lag`):

```{r}
t3 <- train %>%
  select(class, before) %>%
  mutate(before2 = lead(train$before, 1),
         before3 = lead(train$before,2),
         class2 = lead(train$class, 1),
         class3 = lead(train$class, 2),
         transformed = c(train$transformed[-1], NA),
         after = c(train$after[-1], 0)) %>%
  filter(!is.na(before3)) %>%
  rename(class_prev = class, class_next = class3, class = class2,
         before_prev = before, before_next = before3, before = before2) %>%
  select(before_prev, before, before_next, class_prev,
         class, class_next, transformed, after)
```


## Token *class* overview - *previous* vs *next* {.tabset}

We begin this analysis with a tabset of overview plots comparing the frequencies of *classes* at the positions *previous* and *next* in line to all tokens of the tabset *class*. Note the logarithmic x-axes. These plots show the absolute numbers of tokens per *class*. The *classes* are colour-coded for easier comparison. Here is the corresponding helper function:

```{r}
plot_t3_comp <- function(cname){
  p1 <- t3 %>%
    filter(transformed == TRUE & class == cname) %>%
    ggplot(aes(class_prev, fill = class_prev)) +
    labs(x = "Previous class") +
    geom_bar() +
    coord_flip() +
    scale_y_log10() +
    theme(legend.position = "none")

  p2 <- t3 %>%
    filter(transformed == TRUE & class == cname) %>%
    ggplot(aes(class_next, fill = class_next)) +
    labs(x = "Next class") +
    geom_bar() +
    coord_flip() +
    scale_y_log10() +
    theme(legend.position = "none")
  
  layout <- matrix(c(1,2),2,1,byrow=TRUE)
  multiplot(p1, p2, layout=layout)
}
```


### "PLAIN" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 23", out.width="100%"}
plot_t3_comp("PLAIN")
```


### "DATE" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 24", out.width="100%"}
plot_t3_comp("DATE")
```


### "CARDINAL" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 25", out.width="100%"}
plot_t3_comp("CARDINAL")
```


### "MEASURE" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 26", out.width="100%"}
plot_t3_comp("MEASURE")
```


### "ORDINAL" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 27", out.width="100%"}
plot_t3_comp("ORDINAL")
```


### "DECIMAL" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 28", out.width="100%"}
plot_t3_comp("DECIMAL")
```

### "MONEY" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 29", out.width="100%"}
plot_t3_comp("MONEY")
```


### "DIGIT" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 30", out.width="100%"}
plot_t3_comp("DIGIT")
```


### "TELEPHONE" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 31", out.width="100%"}
plot_t3_comp("TELEPHONE")
```


### "TIME" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 32", out.width="100%"}
plot_t3_comp("TIME")
```


### "FRACTION" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 33", out.width="100%"}
plot_t3_comp("FRACTION")
```


### "ADDRESS" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 34", out.width="100%"}
plot_t3_comp("ADDRESS")
```


### "LETTERS" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 35", out.width="100%"}
plot_t3_comp("LETTERS")
```


### "VERBATIM" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 36", out.width="100%"}
plot_t3_comp("VERBATIM")
```


### "ELECTRONIC" {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 37", out.width="100%"}
plot_t3_comp("ELECTRONIC")
```


## Treemap overviews

For an alternative comprehensive overview of the neighbour *class* statistics here are two *treemaps* build using the `treemapify` [package](https://cran.r-project.org/web/packages/treemapify/index.html).

The treemaps summarise at a glance which neighbour combinations exist and are most frequent:

```{r  split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 38", out.width="100%"}
t3 %>%
  group_by(class, class_prev) %>%
  count() %>%
  ungroup() %>%
  mutate(n = log10(n+1)) %>%
  ggplot(aes(area = n, fill = class_prev, label = class_prev, subgroup = class)) +
  geom_treemap() +
  geom_treemap_subgroup_border() +
  geom_treemap_subgroup_text(place = "centre", grow = T, alpha = 0.5, colour =
                             "black", fontface = "italic", min.size = 0) +
  geom_treemap_text(colour = "white", place = "topleft", reflow = T) +
  theme(legend.position = "null") +
  ggtitle("Previous classes grouped by token class; log scale frequencies")
```


```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 39", out.width="100%"}
t3 %>%
  group_by(class, class_next) %>%
  count() %>%
  ungroup() %>%
  mutate(n = log10(n+1)) %>%
  ggplot(aes(area = n, fill = class_next, label = class_next, subgroup = class)) +
  geom_treemap() +
  geom_treemap_subgroup_border() +
  geom_treemap_subgroup_text(place = "centre", grow = T, alpha = 0.5, colour =
                             "black", fontface = "italic", min.size = 0) +
  geom_treemap_text(colour = "white", place = "topleft", reflow = T) +
  theme(legend.position = "null") +
  ggtitle("Next classes grouped by token class; log scale frequencies")
```

The first plot shows the frequency of *previous* token *classes* and the second treemap the frequency of *next* token *classes* (all labeled in white) for each target token *class* (labeled in a large black font and separated by group with grey borders). We use a again a logarithmic frequency scaling to improve the visibility of the rare combinations. Group sizes decrease from the bottom left to the top right of the plot and each subgroup box. The colours of the white-labeled neighbour boxes are identical troughout the plot (e.g. "PUNCT" is always purple). Note, that the subgroup boxes in the two plots have different sizes for identical *classes* because of the `log(n+1)` transformation.


## Relative percentages

### All tokens

Based on these raw numbers we can study the relative contributions of a certain *class* to the *previous* or *next* tokens of another class. To visualise these dependencies, we again determine the `log10(n+1)` frequency distributions for each *class* among the *previous*/*next* tokens depending on the *class* of the reference token. These are the numbers in the bar plots above. We then normalise the range of these numbers (i.e. the height of the bars) to the interval `[0,1]` for each class. This data wrangling is done in the following code block:

```{r}
prev_stat <- t3 %>%
  count(class, class_prev) %>%
  mutate(n = log10(n+1)) %>%
  group_by(class) %>%
  summarise(mean_n = mean(n),
            max_n = max(n),
            min_n = min(n))

next_stat <- t3 %>%
  count(class, class_next) %>%
  mutate(n = log10(n+1)) %>%
  group_by(class) %>%
  summarise(mean_n = mean(n),
            max_n = max(n),
            min_n = min(n))

t3_norm_prev <- t3 %>%
  count(class, class_prev) %>%
  mutate(n = log10(n+1)) %>%
  left_join(prev_stat, by = "class") %>%
  mutate(frac_norm = (n-min_n)/(max_n - min_n),
         test1 = max_n - min_n,
         test2 = n - min_n)

t3_norm_next <- t3 %>%
  count(class, class_next) %>%
  mutate(n = log10(n+1)) %>%
  left_join(next_stat, by = "class") %>%
  mutate(frac_norm = (n-min_n)/(max_n - min_n),
         test1 = max_n - min_n,
         test2 = n - min_n)
```


The result is displayed in two *tile plots* for the *class* vs *previous class* and *class* vs *next class*, respectively. Here, each tile shows the relative frequency (by *class*) of a specific neighbour pairing. The colour coding assigns bluer colours to lower frequencies and redder colours to higher frequencies:

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 40", out.width="100%"}
t3_norm_prev %>%
  ggplot(aes(class, class_prev, fill = frac_norm)) +
  geom_tile() +
  theme(axis.text.x  = element_text(angle=45, hjust=1, vjust=0.9)) +
  scale_fill_distiller(palette = "Spectral") +
  labs(x = "Token class", y = "Previous token class", fill = "Rel freq") +
  ggtitle("Class vs previous class; relative log scale frequencies")
```


```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 41", out.width="100%"}
t3_norm_next %>%
  ggplot(aes(class, class_next, fill = frac_norm)) +
  geom_tile() +
  theme(axis.text.x  = element_text(angle=45, hjust=1, vjust=0.9)) +
  scale_fill_distiller(palette = "Spectral") +
  labs(x = "Token class", y = "Next token class", fill = "Rel freq") +
  ggtitle("Class vs next class; relative log scale frequencies")
```

We find:

- Not all possible neighbouring combinations exist: only about 77% of the tile plots are filled. Certain potential pairs such as "ORDINAL" and "FRACTION" or "TIME" and "MONEY" are never found next to each other.

- "PUNCT" and "PLAIN", the overall most frequent *classes*, also dominate the relative frequencies of *previous*/*next* classes for every token *class*. "PUNCT" is relatively weakly related to "MONEY" (*previous*) and "ORDINAL" (*next*).

- "DIGIT" tokens are often preceded or followed by "LETTERS" tokens.

- "TELEPHONE" is very likely to be preceded by "LETTERS", as well.

- "VERBATIM" tokens are very likely to be followed and preceded by other "VERBATIM" tokens. No other token *class* has such a strong correlation with itself. Some, such as "DIGIT", "MONEY", or "TELEPHONE" are rather unlikely to be preceded by the same *class* 

- "VERBATIM" is also very unlikely to be preceded by "ADDRESS" and "TIME" or followed by "ADDRESS".


### Transformed tokens only

Now we restrict this neighbour analysis to the *transformed* tokens only. Naturally, these will still have transformed or untransformed tokens in the *previous* and *next* positions. Having set up our "context" data frame with this option in mind, we only need to modify our code slightly to prepare the corresponding tile plots:

```{r}
prev_stat <- t3 %>%
  filter(transformed == TRUE) %>%
  count(class, class_prev) %>%
  mutate(n = log10(n+1)) %>%
  group_by(class) %>%
  summarise(mean_n = mean(n),
            max_n = max(n),
            min_n = min(n))

next_stat <- t3 %>%
  filter(transformed == TRUE) %>%
  count(class, class_next) %>%
  mutate(n = log10(n+1)) %>%
  group_by(class) %>%
  summarise(mean_n = mean(n),
            max_n = max(n),
            min_n = min(n))

t3_norm_prev <- t3 %>%
  filter(transformed == TRUE) %>%
  count(class, class_prev) %>%
  mutate(n = log10(n+1)) %>%
  left_join(prev_stat, by = "class") %>%
  mutate(frac_norm = (n-min_n)/(max_n - min_n),
         test1 = max_n - min_n,
         test2 = n - min_n)

t3_norm_next <- t3 %>%
  filter(transformed == TRUE) %>%
  count(class, class_next) %>%
  mutate(n = log10(n+1)) %>%
  left_join(next_stat, by = "class") %>%
  mutate(frac_norm = (n-min_n)/(max_n - min_n),
         test1 = max_n - min_n,
         test2 = n - min_n)
```

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 42", out.width="100%"}
t3_norm_prev %>%
  ggplot(aes(class, class_prev, fill = frac_norm)) +
  geom_tile() +
  theme(axis.text.x  = element_text(angle=45, hjust=1, vjust=0.9)) +
  scale_fill_distiller(palette = "Spectral") +
  labs(x = "Token class", y = "Previous token class", fill = "Rel freq") +
  ggtitle("Class vs previous class; relative log scale; transformed")
```

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 43", out.width="100%"}
t3_norm_next %>%
  ggplot(aes(class, class_next, fill = frac_norm)) +
  geom_tile() +
  theme(axis.text.x  = element_text(angle=45, hjust=1, vjust=0.9)) +
  scale_fill_distiller(palette = "Spectral") +
  labs(x = "Token class", y = "Next token class", fill = "Rel freq") +
  ggtitle("Class vs next class; relative log scale; transformed")
```

We find:

- The *class* "PUNCT" is now gone from the "Token class" axis, since those symbols are never transformed.

- Two combinations disappear: "VERBATIM" is not transformed if preceeded or followed by "TIME" (relatively infrequent neighbours to start with).

- The pair of "PLAIN" preceded by "CARDINAL" significantly increases in frequency (within the rarely transformed "PLAIN" *class*).

- In general, the *classes* "PLAIN" and "VERBATM" are experiences the most visible changes with respect to the total set of neighbouring tokens since these are the *classes* with the highest percentage of untransformed tokens (after "PUNCT", of course).


### No cross-sentence pairs

The previous plots did not distinguish between neighbouring tokens that were placed at the end of one sentence and the beginning of another. Since the sentences in our data set are unrelated and in a random order, the end of one sentence should not influence the beginning of the next one. Here we take this into account by removing those pair of tokens that bridge two sentences.

```{r}
foo <- train %>%
  group_by(sentence_id) %>%
  summarise(sentence_len = max(token_id)) %>%
  ungroup()

bar <- train %>%
  left_join(foo, by = "sentence_id") %>%
  mutate(first_token = token_id == 0,
         last_token = token_id == sentence_len) %>%
  slice(c(-1,-nrow(train))) %>%
  select(first_token, last_token)

prev_stat <- t3 %>%
  bind_cols(bar) %>%
  filter(first_token == FALSE) %>%
  count(class, class_prev) %>%
  mutate(n = log10(n+1)) %>%
  group_by(class) %>%
  summarise(mean_n = mean(n),
            max_n = max(n),
            min_n = min(n))

next_stat <- t3 %>%
  bind_cols(bar) %>%
  filter(last_token == FALSE) %>%
  count(class, class_next) %>%
  mutate(n = log10(n+1)) %>%
  group_by(class) %>%
  summarise(mean_n = mean(n),
            max_n = max(n),
            min_n = min(n))

t3_norm_prev <- t3 %>%
  bind_cols(bar) %>%
  filter(first_token == FALSE) %>%
  count(class, class_prev) %>%
  mutate(n = log10(n+1)) %>%
  left_join(prev_stat, by = "class") %>%
  mutate(frac_norm = (n-min_n)/(max_n - min_n),
         test1 = max_n - min_n,
         test2 = n - min_n)

t3_norm_next <- t3 %>%
  bind_cols(bar) %>%
  filter(last_token == FALSE) %>%
  count(class, class_next) %>%
  mutate(n = log10(n+1)) %>%
  left_join(next_stat, by = "class") %>%
  mutate(frac_norm = (n-min_n)/(max_n - min_n),
         test1 = max_n - min_n,
         test2 = n - min_n)
```

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 40", out.width="100%"}
t3_norm_prev %>%
  ggplot(aes(class, class_prev, fill = frac_norm)) +
  geom_tile() +
  theme(axis.text.x  = element_text(angle=45, hjust=1, vjust=0.9)) +
  scale_fill_distiller(palette = "Spectral") +
  labs(x = "Token class", y = "Previous token class", fill = "Rel freq") +
  ggtitle("Class vs previous class; relative log scale; no sentence bridging")
```


```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 41", out.width="100%"}
t3_norm_next %>%
  ggplot(aes(class, class_next, fill = frac_norm)) +
  geom_tile() +
  theme(axis.text.x  = element_text(angle=45, hjust=1, vjust=0.9)) +
  scale_fill_distiller(palette = "Spectral") +
  labs(x = "Token class", y = "Next token class", fill = "Rel freq") +
  ggtitle("Class vs next class; relative log scale; no sentence bridging")
```

We find:

- The differences to the original plot are only marginal; possibly because of the relatively small contribution of cross-sentence neighbours to the overall sample. With about 750k sentences and close to 10 million tokens there are about 7.5% of pairs bridging two sentences.

- In the *class vs next class* plot the differences are exclusively seen in the "PUNCT" *class*. This is to be expected since a full stop (".") belongs to the "PUNCT" category. Thus, we can confirm the expectation that (practically) all sentences end like this one here.


Actually, let's see if this is true. Here we select only the final tokens of each sentence and plot their *class* distribution. We also plot the frequency of the the different tokens within the "PUNCT" *class*. Note the logarithmic axes:

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 42", out.width="100%"}
bar <- train %>%
  left_join(foo, by = "sentence_id") %>%
  filter(token_id == sentence_len)

p1 <- bar %>%
  ggplot(aes(class, fill = transformed)) +
  geom_bar(position = "dodge") +
  scale_y_log10() +
  ggtitle("Final tokens of sentence")

p2 <- bar %>%
  filter(class == "PUNCT") %>%
  ggplot(aes(before, fill = before)) +
  geom_bar() +
  scale_y_log10() +
  theme(legend.position = "none", axis.text.x = element_text(face = "bold", size = 16)) +
  ggtitle("Final PUNCT tokens")

layout <- matrix(c(1,2),1,2,byrow=TRUE)
multiplot(p1, p2, layout=layout)
```

We find:

- The "PUNCT" class is indeed the most frequent by far, but four other *classes* are found at the end of a sentence, too: "PLAIN", "LETTERS", "DATE", and "MONEY". This could already be kind-of seen in Fig. 5 above.

- As we already know, no "PUNCT" tokens were transformed. All "DATE", "LETTERS", and "MONEY" tokens were transformed, which is not self-evident (see Fig. 5 and the numbers below it).

- For the "PLAIN" *class*, about half of the tokens were transformed, which is a way larger fraction than for the overall "PLAIN" sample. This is an interesting result.

- There are only two cases in which the last token belongs to the "MONEY" *class*:

```{r}
bar %>% filter(class == "MONEY") %>% select(-transformed, -sentence_len)
```

You would be absolutely right in thinking that these look suspicious. Let's see what they really are:

```{r}
before_vs_after(37375)
```

```{r}
before_vs_after(590317)
```

And indeed we have two references to the moderately successful TV show "Numb3rs" ([IMDB](http://www.imdb.com/title/tt0433309/)), whose characters were apparently engaging in the kind of mathematical detective work you'll find much more successfully done in many Kaggle kernels ;-) . No rupees here, I'm afraid. Although, why exactly in our data there is a space in between the two parts of that name is not entirely clear to me.


## Explore all neighbour relations {.tabset .tabset-fade .tabset-pills}

To round off this section, we will create a set of interactive 3D plots using the `plotly` [package](https://cran.r-project.org/web/packages/plotly/index.html). Here, you are able to explore the parameter space of neighbouring *classes*. The grid is defined as *previous class* (x), *next class* (y), and *token class* (z) and the corresponding frequencies are indicated by the colour and size of the data points.


### All tokens {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 43", out.width="100%"}
t3 %>%
  group_by(class, class_prev, class_next) %>%
  count() %>%
  ungroup() %>%
  mutate(class = as.character(class),
         class_prev = as.character(class_prev),
         class_next = as.character(class_next)) %>%
  arrange(desc(n)) %>%
  plot_ly(x = ~class_prev, y = ~class_next, z = ~class, color = ~log10(n),
          text = ~paste('Class:', class,
                        '<br>Previous class:', class_prev,
                        '<br>Next class:', class_next,
                        '<br>Counts:', n)) %>%
  add_markers(size = ~log10(n)) %>%
  layout(title = "Class Neighbour frequencies",
         scene = list(xaxis = list(title = 'Previous Class'),
                     yaxis = list(title = 'Next Class'),
                     zaxis = list(title = 'Class')))
```

### Transformed tokens {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 44", out.width="100%"}
t3 %>%
  filter(transformed == TRUE) %>%
  group_by(class, class_prev, class_next) %>%
  count() %>%
  ungroup() %>%
  mutate(class = as.character(class),
         class_prev = as.character(class_prev),
         class_next = as.character(class_next)) %>%
  arrange(desc(n)) %>%
  plot_ly(x = ~class_prev, y = ~class_next, z = ~class, color = ~log10(n),
          text = ~paste('Class:', class,
                        '<br>Previous class:', class_prev,
                        '<br>Next class:', class_next,
                        '<br>Counts:', n)) %>%
  add_markers(size = ~log10(n)) %>%
  layout(title = "Class Neighbour frequencies",
         scene = list(xaxis = list(title = 'Previous Class'),
                     yaxis = list(title = 'Next Class'),
                     zaxis = list(title = 'Class')))
```


### No cross-sentence pairs {-}

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 45", out.width="100%"}
foo <- train %>%
  group_by(sentence_id) %>%
  summarise(sentence_len = max(token_id)) %>%
  ungroup()

bar <- train %>%
  left_join(foo, by = "sentence_id") %>%
  mutate(first_token = token_id == 0,
         last_token = token_id == sentence_len) %>%
  slice(c(-1,-nrow(train))) %>%
  select(first_token, last_token)

t3 %>%
  bind_cols(bar) %>%
  filter(first_token == FALSE & last_token == FALSE) %>%
  group_by(class, class_prev, class_next) %>%
  count() %>%
  ungroup() %>%
  mutate(class = as.character(class),
         class_prev = as.character(class_prev),
         class_next = as.character(class_next)) %>%
  arrange(desc(n)) %>%
  plot_ly(x = ~class_prev, y = ~class_next, z = ~class, color = ~log10(n),
          text = ~paste('Class:', class,
                        '<br>Previous class:', class_prev,
                        '<br>Next class:', class_next,
                        '<br>Counts:', n)) %>%
  add_markers(size = ~log10(n)) %>%
  layout(title = "Class Neighbour frequencies",
         scene = list(xaxis = list(title = 'Previous Class'),
                     yaxis = list(title = 'Next Class'),
                     zaxis = list(title = 'Class')))
```


## Summary

Certain combinations of classes are more likely to be found next to one another than other combinations. Ultimately, this reflects the grammar structure of the language.

By making use of these next-neighbour statistics we can estimate the probability that a token was classified correctly by iteratively cross-checking the other tokens in the same sentence. **This adds a certain degree of context to a classification/normalisation attempt that is only considering the token itself.**


# Transformation statistics

This section will look at specific sentence parameters and how they affect the transformation statistics or behaviour.


## Sentence length and classes

We begin by studying how the *length* (in tokens) of the sentences affects the statistics of *classes* and transformed tokens. For this we estimate the mean transformed fraction for each group of sentences with the same length together with the corresponding uncertainties. In addition, we examine the *class* frequencies for the shortest sentences and the sentence lenth distributions for each *class*: 

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 46", out.width="100%"}
foo <- train %>%
  group_by(sentence_id, transformed) %>%
  count() %>%
  spread(transformed, n, fill = 0) %>%
  mutate(frac = `TRUE`/(`TRUE` + `FALSE`),
         sen_len = `TRUE` + `FALSE`)
  
bar <- foo %>%
  group_by(sen_len) %>%
  summarise(mean_frac = mean(frac),
            sd_frac = sd(frac),
            ct = n())

foobar <- foo %>%
  left_join(train, by = "sentence_id")

p1 <- bar %>%
  ggplot(aes(sen_len, mean_frac, size = ct)) +
  geom_errorbar(aes(ymin = mean_frac-sd_frac, ymax = mean_frac+sd_frac),
                width = 0., size = 0.7, color = "gray30") +
  geom_point(col = "red") +
  scale_x_log10() +
  labs(x = "Sentence length", y = "Average transformation fraction")

p2 <- foobar %>%
  filter(sen_len < 6) %>%
  group_by(class, sen_len) %>%
  count() %>%
  ungroup() %>%
  filter(n > 100) %>%
  ggplot(aes(sen_len, n, fill = class)) +
  geom_col(position = "fill") +
  scale_fill_brewer(palette = "Set1") +
  labs(x = "Sentence length", y = "Proportion per class")

p3 <- foobar %>%
  ggplot(aes(sen_len)) +
  geom_density(bw = .1, size = 1.5, fill = "red") +
  scale_x_log10() +
  labs(x = "Sentence length") +
  facet_wrap(~ class, nrow = 2)
  
layout <- matrix(c(1,2,3,3),2,2,byrow=TRUE)
multiplot(p1, p2, p3, layout=layout)
p1 <- 1; p2 <- 1; p3 <- 1
```

We find:

- For sentences with more than about 10 tokens the proportion of transformed tokens first decreases somewhat up to about 20 and then increases slightly afterwards. However, none of the changes appear to be significant within their standard deviations. Here, the size of the red data points is proportional to the number of cases per group.

- Interestingly, for very short sentences of only 2 or 3 tokens the mean fraction of transformed tokens is significantly higher: about 33% for 3 tokens and ** practically always** 50% for 2 tokens.

- Upon closer inspection in the bar plot we find that **almost all 2-token sentences consist of a "DATE" *class* and a "PUNCT" *class* token.** Note that our plot omits rare *classes* with less than 100 cases for better visibility. Below is the complete table for the 2-token sentences.

- The 3-token sentences basically just add a "PLAIN" token to the mix, reducing the proportions of "DATE" and "PUNCT" tokens to 1/3 each. For longer sentences, the mix starts to become more heterogenenous. Interestingly, for longer sentences the "PUNCT" fraction does not continue to decline geomtrically - indicating the presence of tokens other than the final full stop.

- Among the individual *classes* we can see considerable differences in the shape of their sentence length distributions. For instance, "VERBATIM" has a far broader distribution that most; reaching notable number above 100 tokens. "PLAIN", "MEASURE", and "DECIMAL" have the sharpest peaks, while "DATE" has an interesting step structure in addition to the dominance in short sentences that we discussed above (here is the promised table for 2-token sentences:)


```{r}
foobar %>%
  filter(sen_len <= 2) %>%
  group_by(sen_len, class, transformed) %>%
  count()
```


## Normalised tokens and where to find them

With apologies to J.K. Rowling we embark upon a quest to determine whether the position of a token in a sentence reveals something about whether it will be transformed.

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 47", out.width="100%"}
foo <- train %>%
  group_by(sentence_id) %>%
  summarise(sentence_len = max(token_id))

p1 <- train %>%
  ggplot(aes(token_id, fill = transformed)) +
  geom_density(bw = .1, alpha = 0.5) +
  scale_x_log10() +
  theme(legend.position = "none")

p2 <- train %>%
  left_join(foo, by = "sentence_id") %>%
  mutate(token_rel = token_id/sentence_len) %>%
  ggplot(aes(token_rel, fill = transformed)) +
  geom_density(bw = .05, alpha = 0.5) +
  labs(x = "Relative token position")

p3 <- train %>%
  left_join(foo, by = "sentence_id") %>%
  mutate(token_rel = token_id/sentence_len) %>%
  ggplot(aes(token_rel, fill = transformed)) +
  geom_density(bw = .05, alpha = 0.5) +
  labs(x = "Relative token position") +
  facet_wrap(~class, nrow = 2) +
  theme(legend.position = "none")

layout <- matrix(c(1,2,1,2,3,3,3,3,3,3),5,2,byrow=TRUE)
multiplot(p1, p2, p3, layout=layout)
p1 <- 1; p2 <- 1; p3 <- 1

```

We find:

- Tokens around absolute positions of 10 have notably fewer transformations that others. We also the effect of the 50% transformations for 2-token sentences. There is a small effect for the highest *token\_ids* but it doesn't look significant.

- In terms of relative position we find that transformed tokens are more likely to be found in the second half of a sentence compared to the first half. An exception is the very end of the sentence where we most likely see the influence of the untransformed punctuation. It is interesting how much more evenly distributed the non-transformed tokens are.

- By facetting the second plot by *class* recover some of the information we had seen in the boxplots above, but we can also clearly see the impact of the normalisation on the individual *classes*. The most interesting distributions are clearly "ELECTRONIC", more likely to transformed early and late in a sentence, and "MONEY", practically the opposite. But also "VERBATIM" and "MEASURE" have interesting features.


# Feature engineering

We have already introduced the derived *sentence length* and *transformed* features. Now we want to engineer a few more variables that might be useful for our understanding of the data and our ultimate prediction goal. All features will be defined in the single code block below, which will likely grow as we design more of them. The following subsections will examine these features and their impact on the token normalisation.


```{r}
train <- train %>%
  rowid_to_column(var = "row_id")

# class-specific distances
#---
classes <- train %>%
  select(class) %>%
  unique() %>%
  unlist(use.names = FALSE) %>%
  as.character()

cl_dist <- NULL
for (i in classes){
  
  cl_dist <- train %>%
    filter(class == !!quo(i)) %>%
    mutate(cl_dist = pmax(0,lead(token_id,1)-token_id)*pmax(0,sentence_id-lead(sentence_id,1)+1)) %>%
    select(row_id, cl_dist) %>%
    bind_rows(cl_dist)
}
cl_dist <- cl_dist %>%
  mutate(cl_dist = as.integer(cl_dist)) %>%
  replace_na(list(cl_dist = 0)) %>%
  arrange(row_id)
#---

sen_len <- train %>%
  group_by(sentence_id) %>%
  summarise(sentence_len = max(token_id))

train <- train %>%
  mutate(str_len = str_length(before),
         cl_dist = cl_dist$cl_dist,
         num = !is.na(as.numeric(before))
         ) %>%
  left_join(sen_len, by = "sentence_id")
```


## String length

We start with a pretty basic feature: the length of the *before* token string in numbers of characters. These are the overall distribution, the impact on the "transformed" status, and the string lengths by class:

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 47", out.width="100%"}
p1 <- train %>%
  ggplot(aes(str_len)) +
  geom_histogram(fill = "red", bins = 50) +
  scale_x_log10() +
  scale_y_log10()

p2 <- train %>%
  ggplot(aes(str_len, fill = transformed)) +
  geom_density(alpha = 0.5, bw = 0.1) +
  scale_x_log10()

p3 <- train %>%
  ggplot(aes(class, str_len, col = class)) +
  geom_boxplot() +
  scale_y_log10() +
  theme(legend.position = "none", axis.text.x  = element_text(angle=45, hjust=1, vjust=0.9))

layout <- matrix(c(1,2,3,3),2,2,byrow=TRUE)
multiplot(p1, p2, p3, layout=layout)
p1 <- 1; p2 <- 1; p3 <- 1
```

We find:

- Short strings are dominant in our data (note the log axes) but longer ones with up to a few 100 characters can occur.

- The transformed vs untransformed string lengths reveal an interesting multi-modal distribution where transformed strings are likely to be very long or just a few characters long. Very short strings are more likely to be untransformed, which might be due to the impact of the "PUNCT" class since punctuation usually is only a single character.

- Looking closer at the boxplots of *string length* per *class* there are significant differences. We confirm that "PUNCT" tokens are typically short, but the same appears to be true for "VERBATIM" tokens. "ELECTRONIC" tokens make up the majority of the longest tokens and, together with "TELEPHONE", are the only ones with a median above 10 characters.

From these plots, it's definitely worth it to examine the transformed vs untransformed statistics in more detail. Here we split the corresponding density plots by *class*:


```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 48", out.width="100%"}
train %>%
  ggplot(aes(str_len, fill = transformed)) +
  geom_density(alpha = 0.5, bw = 0.1) +
  scale_x_log10() +
  facet_wrap(~ class)
```

We find:

- Most classes are always transformed and we see the difference in their overall string length distribution; e.g. "TELEPHONE" peaks just above 10 vs "TIME" peasking below 10, or "DIGIT", "ADDRESS", "CARDINAL", and "FRACTION" tokens being predominantly shorter than 10 characters. "PUNCT" is short and never transformed.

- There's a really interesting bimodality in the "ELECTRONIC" class, with short tokens never transformed and longer tokens always transformed. The overlap between the two populations appears to be marginal. The same observation is true for the "LETTERS" class, albeit with a larger overlap.

- A similar effect, although less pronounced, can be seen in the "MEASURE" class, where untransformed tokens are on average longer than transformed ones. For "VERBATIM" we see that longer tokens are always transformed.


## Class-specific token distance

Next, we define a measure that we call the *class-specific token distance*. What we mean by that is simply the number of tokens it takes to get from one specific token to the next token of the same *class*. For instance, if our sentence goes like "DATE PLAIN PLAIN DATE PUNCT" then our *distances* would be "3 1 0 0 0": It takes 3 steps to get from the first "DATE" token to the next "DATE" token, and 1 step to get from the first "PLAIN" token to the one right next to it. The last three tokens don't have another token of the same *class* following them **in the same sentence**, so we assign a distance of zero.

The aim here is to get an idea of how the different *classes* cluster. Are "LETTER" tokens more likely to follow each other? "ORDINAL" tokens more often far apart? Knowing these patterns could help us in optimising the *classification* of a sentence and thereby its normalisation patterns.

We start out with a bar plot that compares how often a *class* occurs alone in a sentence (or as the last token of its kind), resulting in a zero distance, versus how often the *class* can be found multiple times in addition to that last token. Here, two tokens per *class* per sentence would give us a 50/50 split: 

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 49", out.width="100%"}
train %>%
  ggplot(aes(cl_dist==0, fill = cl_dist==0)) +
  geom_bar() +
  theme(legend.position = "none") +
  scale_y_log10() +
  facet_wrap(~class) +
  labs(x = "Alone or the last token of its kind in a sentence")
```

We find:

- Most *classes* are predominantly alone in their sentences, except for "PLAIN", "PUNCT", and "VERBATIM". In particular the "PLAIN" tokens are likely to be clustered together. Note the logarithmic y-axes.


The next plots will examine the *class distance* distributions for the different *classes*; ignoring the zero-distance tokens:

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 50", out.width="100%"}
train %>%
  filter(cl_dist > 0) %>%
  ggplot(aes(cl_dist, fill = transformed)) +
  geom_density(bw=0.1, alpha=0.5) +
  scale_x_log10() +
  facet_wrap(~class)

```

We find:

- In terms of *transformed* vs *untransformed* there are notable differences within the "MEASURE", "LETTERS", "ELECTRONIC", and "PLAIN" classes. In particular the "MEASURE" tokens have a much broader distribution when transformed.

- Overall, the "VERBATIM" and "PLAIN" tokens clearly cluster together. Also the "MEASURE" and "TELEPHONE" tokens occur predominantly at short distances to each other. A different shape can be seen in "TIME", "MONEY", and "DECIMAL" which all show a bi-modal distribution with a notable peak near 10 tokens distance.


## Numeric Tokens

This is not at all a sophisticated feature but it might be helpful for something. Here we test if we can successfully convert our tokens into R's *numeric* class. This only works for characters that are formatted like numbers; for everything else the conversion returns `NA`. We record the success of the conversion in the new *num* feature.

Below, we plot the fraction of successfully converted tokens per class. We choose a logarithmic scale to make the magnitudes comparable. This also has the advantage that all bars with less than 1% success are pointing downwards:

```{r split=FALSE, fig.align = 'default', warning = FALSE, fig.cap ="Fig. 51", out.width="100%"}
train %>%
  group_by(class, num) %>%
  count() %>%
  spread(num, n, fill = 0) %>%
  mutate(frac_num = `TRUE`/(`TRUE`+`FALSE`)*100) %>%
  filter(frac_num > 0) %>%
  ggplot(aes(reorder(class, -frac_num, FUN = min), frac_num, fill = class)) +
  geom_col() +
  geom_text(aes(reorder(class, -frac_num, FUN = min), frac_num, label = sprintf("%2.2f", frac_num))) + 
  scale_y_log10() +
  labs(x = "Class", y = "Amount of numerical tokens [%]") +
  theme(legend.position = "none", axis.text.x  = element_text(angle=45, hjust=1, vjust=0.9))

```

We find:

- The high numeric fraction in the "CARDINAL", "DIGIT", and "DECIMAL" classes are not surprising; although it's noteworthy that none of them reaches 100% (average around 90%)

- More interesting is that the majority (57%) of "DATE" tokens are numeric, together with a notable chunk (6%) of "TELEPHONE" tokens. Even a few "TIME" tokens (1%) can be converted into a numeric format.

- There are rare occurences of numeric tokens in the "VERBATIM", "LETTERS", and "PLAIN" *classes*, but they don't come close to even 0.1%.

---

To be continued.