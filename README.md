# sp1541-nlp
### NLP Project using past essays from SP1541: Exploring Science Communication through Popular Science in Academic Year 2020/21 Semester 1

For context, I took a module in Academic Year 2020/21 Semester 1 - SP1541: Exploring Science Communication in Popular Science, where I had to submit 2 news articles for grading.

The first article, titled *Timing vaccination campaign to reduce measles infections* - is related to my academic discipline, and revolves mainly around mathematics.
    
The second article, titled *Investigating the relationship between culture and sweet-sour taste interactions* - is not related to my academic discipline, and is based on the science of chemistry.

Unfortunately, I scored below average for both articles, as I presumed that as a freshman back then, I did not undergo sufficient training to communicate complex scientific concepts well to the layman audience.

With the introduction of ChatGPT however, I took this opportunity to see if this AI tool could optimise my initial write-ups. The following articles/texts will hence be used for this analysis, as described below:

    | Text_id  | Description                            |
    |----------|----------------------------------------|
    | 1a       | News Article 1 - Original              |
    | 1b       | News Article 1 - Optimised (Min)       |
    | 1c       | News Article 1 - Optimised (Max)       |
    | 2a       | News Article 2 - Original              |
    | 2b       | News Article 2 - Optimised (Min)       |
    | 2c       | News Article 2 - Optimised (Max)       |

For submission, the word limits of the 2 articles are 800 and 1000 respectively. For each article, 2 other variants were produced, namely:
- "b" series - using ChatGPT to summarise the original article with as few words as possible (~400 words)
- "c" series - using ChatGPT to stick to the original word limit(s), while enhancing the language and expression of the article text where applicable

Using various libraries in Python including `matplotlib`, `seaborn`, `nltk`, `textstat` and `wordcloud`, we will hence perform detailed comparisons to evaluate if ChatGPT has indeed enhanced or reduced the quality of the original articles.

Three main methods will be used for this analysis:
1. Preliminary analysis - comparing word counts, readability scores and sentiment (compound) scores
2. Creating word clouds to identify most frequently used words from each article
3. Identifying top 10 words within each article series

## Summary of results

### Preliminary analysis
Using ChatGPT in an attempt to optimise the original articles resulted in:
- Decreased in Flesch reading scores (aka readability)
- Slight increase or maintenance of sentiment compound scores (positive tone)

### Top words used among each article series
- Variants of News Article 1: `measles`, `vaccination`, `campaign`, `Pakistan`, `cases`, `infections`, `health`, `November`, `disease`, `children`
- Variants of News Article 2: `taste`, `sweetness`, `sourness`, `sucrose`, `sensitivities`, `study`, `consumers`, `danish`, `acid`, `Chinese`
- Across variants from both articles: `study`, `researchers`, `may`, `one`, `could`, `results`, `2019`

## References

### News Article 1
- Thakkar, N., Gilani, S. S. A., Hasan, Q., & McCarthy, K. A. (2019). Decreasing measles burden by optimizing campaign timing. Proceedings of the National Academy of Sciences, 201818433. Also available from https://www.pnas.org/content/pnas/116/22/11069.full.pdf 
- Patel, M. K., Dumolard, L., Nedelec, Y., Sodha, S. V., Steulet, C., Gacic-Dobo, M., ... & Goodson, J. L. (2019). Progress toward regional measles elimination—worldwide, 2000–2018. Morbidity and Mortality Weekly Report, 68(48), 1105. Also available from https://www.cdc.gov/mmwr/volumes/68/wr/pdfs/mm6848a1-H.pdf 
- National Institute of Population Studies (NIPS) [Pakistan] and ICF. 2019. Pakistan Demographic and Health Survey 2017-18. Islamabad, Pakistan, and Rockville, Maryland, USA: NIPS and ICF. Also available from https://dhsprogram.com/pubs/pdf/FR354/FR354.pdf 
- Pakistan Bureau of Statistics. Block Wise Provisional Summary Results of 6th Population & Housing Census-2017 [As on January 03, 2018]. Also available from http://www.pbs.gov.pk/content/block-wise-provisional-summary-results-6th-population-housing-census-2017-january-03-2018 
- Centers for Disease Control and Prevention. Epidemiology and Prevention of Vaccine-Preventable Diseases. Chapter 10, Measles. 8th Edition, 2004. https://www.cdc.gov/vaccines/pubs/pinkbook/meas.html 

### News Article 2
- Bertino, M., Beauchamp, G. K., & Jen, K. L. C. (1983). Rated taste perception in two cultural groups. Chemical senses, 8(1), 3-15. Also available from https://academic.oup.com/chemse/article-abstract/8/1/3/271785 
- Chamoun, E., Liu, A. A., Duizer, L. M., Darlington, G., Duncan, A. M., Haines, J., & Ma, D. W. (2019). Taste sensitivity and taste preference measures are correlated in healthy young adults. Chemical senses, 44(2), 129-134. Also available from https://pubmed.ncbi.nlm.nih.gov/30590512/ 
- Hydrolysis. (n.d). In Wikipedia. Retrieved from https://en.wikipedia.org/wiki/Hydrolysis 
- Junge, J.Y.; Bertelsen, A.S.; Mielby, L.A.; Zeng, Y.; Sun, Y.-X.; Byrne, D.V.; Kidmose, U. Taste Interactions between Sweetness of Sucrose and Sourness of Citric and Tartaric Acid among Chinese and Danish Consumers. Foods 2020, 9, 1425. Also available from https://www.mdpi.com/2304-8158/9/10/1425 
- Williams, J. A., Bartoshuk, L. M., Fillingim, R. B., & Dotson, C. D. (2016). Exploring ethnic differences in taste perception. Chemical senses, 41(5), 449-456. Also available from https://academic.oup.com/chemse/article/41/5/449/2366044 
