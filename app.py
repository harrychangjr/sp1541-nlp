import streamlit as st
import backend
import matplotlib as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, flesch_kincaid_grade
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd

# Set page title
st.set_page_config(page_title="SP1541-NLP - Harry Chang", page_icon = "desktop_computer", layout = "centered", initial_sidebar_state = "auto")

st.title("NLP Project using Science News Articles")

selected_options = ["Overview", "Articles", "Detailed Walkthrough", "References"]

selected = st.selectbox("What would you like to find out about this project?", options = selected_options)

st.write("Current selection:", selected)

if selected == "Overview":
    st.header("Overview")
    st.markdown("""
    For context, I took a module in Academic Year 2020/21 Semester 1 - SP1541: Exploring Science Communication in Popular Science, where I had to submit 2 news articles for grading.

    The first article, titled *Timing vaccination campaign to reduce measles infections* - is related to my academic discipline, and revolves mainly around mathematics.
    
    The second article, titled *Investigating the relationship between culture and sweet-sour taste interactions* - is not related to my academic discipline, and is based on the science of chemistry.

    Unfortunately, I scored below average for both articles, as I presumed that as a freshman back then, I did not undergo sufficient training to communicate complex scientific concepts well to the layman.

    With the introduction of ChatGPT however, I took this opportunity to see if this AI tool could optimise my initial write-ups. The following articles/texts will hence be used for this analysis, as described below:   
    """)
    st.markdown("""
    | Text_id  | Description                            |
    |----------|----------------------------------------|
    | 1a       | News Article 1 - Original              |
    | 1b       | News Article 1 - Optimised (Min)       |
    | 1c       | News Article 1 - Optimised (Max)       |
    | 2a       | News Article 2 - Original              |
    | 2b       | News Article 2 - Optimised (Min)       |
    | 2c       | News Article 2 - Optimised (Max)       |
    """)
    st.markdown("")
    st.markdown("""
    For submission, the word limits of the 2 articles are 800 and 1000 respectively. For each article, 2 other variants were produced, namely:
    - "b" series - using ChatGPT to summarise the original article with as few words as possible (~400 words)
    - "c" series - using ChatGPT to stick to the original word limit(s), while enhancing the language of the articles

    Using various libraries in Python including `matplotlib`, `seaborn`, `nltk`, `textstat` and `wordcloud`, we will hence perform detailed comparisons to evaluate if ChatGPT has indeed enhanced or reduced the quality of the original articles.

    Three main methods will be used for this analysis:
    1. Preliminary analysis - comparing word counts, readability scores and sentiment (compound) scores
    2. Creating word clouds to identify most frequently used words from each article
    3. Identifying top 10 words within each article series

    The 'Detailed Walkthrough' section will show a summary of the results that were obtained.

    Analysis performed by: [Harry Chang](https://linkedin.com/in/harrychangjr)
    """)
elif selected == "Articles":
    st.header("Articles")
    options = ["Text 1a: News Article 1 - Original", "Text 1b: News Article 1 - Optimised (Min)", "Text 1c: News Article 1 - Optimised (Max)",
            "Text 2a: News Article 2 - Original", "Text 2b: News Article 2 - Optimised (Min)", "Text 2c: News Article 2 - Optimised (Max)"]
    select = st.selectbox("Which article would you like to read?", options = options)
    st.write("Current selection:", select)
    if select == "Text 1a: News Article 1 - Original":
        st.subheader("Timing vaccination campaign to reduce measles infections")
        st.write("*Despite having a vaccine that is readily accessible, measles cases and deaths are still surging worldwide, especially in recent years. Why is this so and are there any long-term solutions to resolve this?*")
        st.write("By: Harry Chang (30 September 2020)")
        st.markdown("""       
        According to an update from the World Health Organisation (WHO), nearly 10 million cases of measles were reported in the year 2018. During that year, more than 140,000 people worldwide have died from the disease. In addition, reported measles cases have surged internationally in devastating outbreaks across different regions.

        Besides the mild symptoms of fever, runny nose and body rashes, this highly contagious disease may also lead to long term effects on the immune systems of those affected. For instance, as many as 1 in 20 children with measles will contract pneumonia, which is the leading cause of death amongst young children. In addition, about 1 in every 1000 who contract measles will develop encephalitis - or brain infection - which can result in the child being deaf or developing intellectual disability. This raises the importance of vaccinating children so that they will not have to live with such complications in the long run.

        Despite the long existence of an effective and cost-efficient vaccine, the outbreak of measles remains a pressing global health issue particularly for developing countries. These nations have often been identified to lack access to vaccinations and high-quality health infrastructure.

        As such, a study on the measles outbreak in Pakistan has predicted that optimising the timing of a vaccination campaign plays an important role in reducing the total infections of measles.

        The study, led by senior researcher Niket Thakkar from the Institute for Disease Modelling (based in the USA), was conducted in response to the sudden increase in measles cases within the span of a year. From 2016 to 2017, the number of cases in Pakistan have more than doubled, as confirmed by local laboratories.

        Prior to the study, measles vaccination coverage in toddlers aged under 2 was estimated to be 61% nationwide, as cited from Pakistan’s Demographic and Health Survey (DHS) in 2012 – 2013. With Pakistan being identified as one of the top countries with the most unvaccinated infants, the need to improve this rate was therefore essential, as suggested by Thakkar and his team of researchers.

        The researchers came up with a mathematical model which uses linear regression to predict the severity of future outbreaks. Using case data from Pakistan that contains the number of new measles cases per month, they predicted the number of cases of subsequent months within the next three years. This data was also categorised by province level to compare the severity of the measles outbreak between different regions in Pakistan.

        To understand how linear regression works, let us think of this example. If you spent \$10 on a Monday, \$20 on a Tuesday, \$30 on a Wednesday, how much would you win on Thursday? If your answer is \$40, you’ve just performed linear regression - this method thus makes use of available information to constantly make predictions.

        This model assisted researchers in understanding when and where the vaccine should be distributed within the country. Their results show that holding a vaccination campaign in November has the greatest impact, with an estimated 440,000 more infections that could be prevented in comparison to a January campaign. These results were later used by the Pakistani government in vaccination planning, which led to the implementation of the campaign in November 2018.

        According to the study, less cases were confirmed from May to October as compared to the rest of the year. This suggests a low transmission season during this period, reiterating why the campaign is best implemented in November, when cases start to surge again. As a result of this implementation, the estimated measles vaccination coverage in infants aged under 2 had improved to 73% nationwide. This statistic was reported in 2017 – 2018’s iteration of Pakistan’s DHS, which was published in January 2019.

        On the other hand, if the campaign was delayed from November 2018 to May 2019, can you guess the number of additional infections that would have occurred? There would have been more than 600,000 additional infections from 2018 to 2021 - this significant number is sandwiched between the population sizes of Sialkot and Sukkur, the 13th and 14th most densely populated cities in Pakistan respectively (out of 99 cities in total). As such, this further justifies the researchers’ preference for the campaign to be conducted in November.

        Beyond immediate outbreak response, countries should continue investing in high quality immunisation programmes, as well as disease surveillance. This would help to ensure that these outbreaks are detected quickly and stopped as soon as possible.

        It is indeed a tragedy to witness a sudden increase in cases and deaths from a disease that is easily preventable, especially in recent times. Therefore, it is crucial to ensure that even the poorest countries have access to these high-quality vaccination programmes. This would help prevent the unnecessary loss of lives to easily treatable diseases, including measles.
        """)       
    elif select == "Text 1b: News Article 1 - Optimised (Min)":
        st.subheader("Optimizing Vaccination Campaign Timing to Reduce Measles Infections")
        st.write("By: ChatGPT (26 April 2023)")
        st.markdown("""
        Despite the availability of an effective vaccine, measles cases and deaths continue to surge worldwide. The World Health Organization (WHO) reported nearly 10 million cases and over 140,000 deaths from measles in 2018. Measles, a highly contagious disease, can cause severe long-term effects on the immune systems of those affected, making vaccination crucial to protect children from lifelong complications.

        Although an effective and cost-efficient vaccine exists, measles outbreaks remain a significant global health issue, particularly in developing countries with limited access to vaccinations and high-quality health infrastructure. A study on the measles outbreak in Pakistan found that optimizing the timing of vaccination campaigns could significantly reduce measles infections.

        Senior researcher Niket Thakkar from the Institute for Disease Modeling (USA) led the study in response to the sudden increase in measles cases in Pakistan from 2016 to 2017. With only 61% of toddlers under two vaccinated, Pakistan has one of the highest numbers of unvaccinated infants worldwide.

        The research team developed a mathematical model using linear regression to predict the severity of future outbreaks. They analyzed case data from Pakistan, categorized by province, and predicted the number of cases for subsequent months within the next three years. This model helped researchers determine when and where to distribute vaccines within the country.

        Results showed that a vaccination campaign in November would have the greatest impact, preventing an estimated 440,000 more infections than a campaign in January. The Pakistani government used these findings to plan their vaccination campaign in November 2018. Consequently, the estimated measles vaccination coverage in infants under two increased to 73% nationwide.

        The study revealed a low transmission season from May to October, which highlights the importance of implementing the campaign in November when cases start to surge again. Delaying the campaign from November 2018 to May 2019 would have resulted in over 600,000 additional infections.

        To prevent future outbreaks, countries must invest in high-quality immunization programs and disease surveillance. Ensuring access to high-quality vaccination programs in even the poorest countries is crucial to prevent unnecessary loss of lives to easily treatable diseases like measles.
        """)
    elif select == "Text 1c: News Article 1 - Optimised (Max)":
        st.subheader("The Importance of Optimized Vaccination Campaign Timing in Reducing Measles Infections")
        st.write("By: ChatGPT (26 April 2023)")
        st.markdown("""
        Despite the availability of an effective vaccine, measles cases and deaths continue to surge worldwide. The World Health Organization (WHO) reported nearly 10 million cases and over 140,000 deaths from measles in 2018. Measles is a highly contagious disease that can cause severe long-term effects on the immune systems of those affected, making vaccination crucial to protect children from lifelong complications. Although an effective and cost-efficient vaccine exists, measles outbreaks remain a significant global health issue, particularly in developing countries with limited access to vaccinations and high-quality health infrastructure.

        One such country facing a measles outbreak is Pakistan, where measles vaccination coverage for toddlers under two years old was estimated at only 61%, according to the country's Demographic and Health Survey (DHS) in 2012-2013. With Pakistan identified as one of the top countries with the most unvaccinated infants, there is an urgent need to improve vaccination rates to curb the measles outbreak.

        Measles can cause severe health complications, including pneumonia, which affects as many as 1 in 20 children with measles and is the leading cause of death among young children. Furthermore, about 1 in every 1,000 children who contract measles will develop encephalitis or brain infection, which can result in deafness or intellectual disability. These potential lifelong complications underscore the importance of vaccinating children against measles.

        In response to the sudden increase in measles cases in Pakistan from 2016 to 2017, senior researcher Niket Thakkar from the Institute for Disease Modeling (USA) led a study to determine whether optimizing the timing of a vaccination campaign could significantly reduce measles infections. The research team developed a mathematical model using linear regression to predict the severity of future outbreaks. They analyzed case data from Pakistan, categorized by province, and predicted the number of cases for subsequent months within the next three years.

        Linear regression is a method that makes use of available information to make predictions. For example, if you spent \$10 on a Monday, \$20 on a Tuesday, and \$30 on a Wednesday, you would predict spending \$40 on Thursday. In the context of the study, the researchers used linear regression to estimate future measles cases based on past case data.

        The mathematical model helped researchers determine when and where to distribute vaccines within the country. Results showed that a vaccination campaign in November would have the greatest impact, preventing an estimated 440,000 more infections than a campaign in January. The Pakistani government used these findings to plan their vaccination campaign in November 2018. Consequently, the estimated measles vaccination coverage in infants under two years old increased to 73% nationwide, as reported in the 2017-2018 iteration of Pakistan's DHS published in January 2019.

        The study revealed a low transmission season from May to October, which highlights the importance of implementing the campaign in November when cases start to surge again. Delaying the campaign from November 2018 to May 2019 would have resulted in over 600,000 additional infections. This number is significant, as it falls between the population sizes of Sialkot and Sukkur, the 13th and 14th most densely populated cities in Pakistan, respectively, out of 99 cities in total.

        The success of the optimized vaccination campaign in Pakistan demonstrates the importance of timing in reducing measles infections. Beyond immediate outbreak response, countries should continue investing in high-quality immunization programs and disease surveillance. This would ensure that outbreaks are detected quickly and stopped as soon as possible. Moreover, effective communication and collaboration between governments, health organizations, and communities are crucial in promoting vaccination awareness and combating vaccine hesitancy.

        It is also essential to address logistical challenges in vaccine distribution, particularly in remote areas and conflict zones. This includes investing in better transportation, storage facilities, and trained healthcare personnel to administer the vaccines. Providing incentives for vaccination, such as conditional cash transfers or linking vaccination to other social services, may further encourage parents to have their children vaccinated.

        The use of technology can also play a significant role in improving vaccination campaigns. For example, mobile phone-based applications can help monitor vaccination coverage, track stock levels, and send reminders to parents about their children's vaccination schedules. Furthermore, the use of Geographic Information Systems (GIS) can help identify areas with low vaccination coverage and target interventions accordingly.

        Global cooperation and commitment are also critical in reducing measles infections worldwide. Developed countries and international organizations should support developing nations in their efforts to improve vaccination coverage by providing financial assistance, technical expertise, and resources. Additionally, multinational vaccination campaigns should be encouraged, as measles outbreaks in one country can quickly spread across borders due to international travel and migration.

        The recent measles outbreaks worldwide underscore the importance of optimizing vaccination campaign timing, as evidenced by the study conducted in Pakistan. By identifying the most effective time for vaccine distribution, countries can significantly reduce measles infections and save lives. However, optimizing vaccination campaign timing is just one aspect of a broader strategy to combat measles and other vaccine-preventable diseases. Continued investment in high-quality immunization programs, disease surveillance, and global cooperation is necessary to ensure that all children have access to life-saving vaccines and the opportunity to live healthy, productive lives.
        """)
    elif select == "Text 2a: News Article 2 - Original":
        st.subheader("Investigating the relationship between culture and sweet-sour taste interactions")
        st.write("*Are we correct to stereotype taste perceptions and preferences based on different cultures?*")
        st.write("By: Harry Chang (31 October 2020)")
        st.markdown("""
        Imagine that you are drinking a glass of margarita. After that first sip, you find that your drink is too sour. You then lick some of the salt from the rim of the glass before taking a second sip. You find that now, the margarita tastes less sour! This is a perfect example of a taste interaction between different taste qualities. 

        When two or more taste qualities interact, they affect the perception of one another. The taste qualities involved can either enhance or suppress one another, which is dependent on their concentrations.

        Previous studies have shown that cultural differences do affect taste sensitivities and taste interactions amongst different individuals. For instance, a US study discovered that taste sensitivities across all five taste qualities are lower amongst individuals of African-American and Hispanic origin compared to Caucasians. Another study also revealed that Taiwanese students tend to have higher sweetness sensitivity compared to their American counterparts.

        Thus, a new study has sought to validate the above-mentioned findings. Conducted by a group of Danish and Chinese researchers earlier this year, this cross-cultural study suggested that culture does affect taste interactions to a certain degree. In particular, Danish consumers experienced a smaller extent of sourness suppression by the sweetness of sucrose as compared to their Chinese counterparts.

        To the researchers’ surprise, they discovered that the vice versa does not prove to be true. As they could not establish a relationship between culture and the extent of sourness suppressing sweetness, this suggests that the differences in taste perception may need to be further classified at the individual level. 

        So, in the first place, how does the suppression between sweetness and sourness occur?

        Sucrose, widely known as table sugar, undergoes hydrolysis to be further decomposed to glucose and fructose. This process can be quickened by introducing acids such as citric acid, commonly present in fruit juices. As such, hydrolysis reduces the formation of sugar crystals, explaining how sourness can suppress sweetness.

        Let us now try to break down this process from another perspective. Citric acid, for instance, has an estimated pH of 2.2, which exemplifies its relatively high acidity to its sourness. On the other hand, sucrose is known to be slightly alkaline – and a bit bitter, given its higher pH value of 8. However, when citric acid is introduced to sucrose, the resultant solution will have a pH range between 4 to 7. Simply put, we can also make use of the same reaction to explain how sweetness suppresses sourness.

        To further appreciate the findings of the study, we should also understand how taste sensitivities relate to taste preferences. 

        In a separate study conducted by Canadian researchers in 2019, there is evidence of a relationship between certain taste sensitivities of consumers and their taste preferences. In particular, sweetness and saltiness were revealed to be less preferred by consumers who recorded higher sensitivities in these two taste qualities.

        As such, the study sought to investigate how these taste interactions vary between both cultures. The Danish and Chinese test subjects evaluated six liquid mixtures: namely water, sucrose, tartaric acid, citric acid, a mixture of sucrose with tartaric acid and sucrose mixed with citric acid. Both citric and tartaric acids were used as samples to exhibit the taste of sourness, while sucrose was used to exhibit sweetness.

        Participants were tasked to taste one sample at a time, with a 30 second break in between after rinsing their mouth with water. For each sample, the taste sensitivity was evaluated on a 9-point scale, with 1 being ‘not at all’ and 9 rated as ‘extremely’ sweet or sour, depending on the sample.

        For samples containing a mixture of sweetness and sourness, they were evaluated with an additional ‘Just About Right’ (JAR) scale to measure the appropriate concentrations of sucrose and acids based on each individual. Together with the taste sensitivities, the JAR ratings for each sample were recorded using a questionnaire. Data collected from this questionnaire was later used for further analysis.

        On average, the Danish consumers consistently recorded higher sweetness sensitivities than their Chinese counterparts. This further explains how the researchers concluded that sucrose had managed to suppress tartaric acid to a greater extent in Chinese consumers compared to the Danish consumers.

        The researchers added that based on their research on similar studies, a Caucasian population generally tends to have a lower taste sensitivity for sweetness, sourness, saltiness and bitterness than an Asian population. Since an inverse relationship between taste sensitivity and taste preference has been established from the Canadian study, comparing the taste sensitivities of these Chinese and Danish consumers may therefore not be entirely representative of the ‘East vs. West’ comparison.

        In addition, the researchers could not conclude if culture had a role in sweetness suppression by sourness. This is because the results had varied between each individual, regardless of whether they were Chinese or Danish.

        Upon obtaining the necessary readings, the test subjects were later divided into three different clusters based on their relative sensitivities to sourness. For instance, consumers with similarly low sourness sensitivity were grouped together under the same category. This may be due to the high suppression of sourness by sweetness.

        This same method of classification based on sweetness was also performed on these customers. Likewise, each cluster comprised of both Chinese and Danish consumers. While there were certain trends that may imply a relationship between culture and taste interactions, the researchers could not affirm this conclusion upon further analysis.

        Overall, beverage manufacturers stand to benefit most from the results of the study. In order to boost their sales revenue, they would need to re-evaluate their product segmentation strategies to diversify their target consumer range. Instead of focusing on culture, these companies may wish to explore other variables such as age and gender instead.

        With this stereotype debunked, do we now expect people of different cultures to appreciate unique drinks such as sugarcane juice with lemon the same way? Only time and experience will tell. 
        """)
    elif select == "Text 2b: News Article 2 - Optimised (Min)":
        st.subheader("Exploring the Connection Between Culture and Sweet-Sour Taste Interactions")
        st.write("*Debunking Stereotypes of Taste Perception and Preferences Across Cultures*")
        st.write("By: ChatGPT (26 April 2023)")
        st.markdown("""
        Imagine sipping a margarita and finding it too sour. After licking salt from the rim, you take another sip and discover it tastes less sour. This exemplifies taste interactions, where two or more taste qualities affect each other's perception, either enhancing or suppressing one another.

        Past research suggests cultural differences impact taste sensitivities and interactions. For example, a US study found individuals of African-American and Hispanic origin have lower taste sensitivities across all five taste qualities than Caucasians. Another study revealed Taiwanese students possess higher sweetness sensitivity than Americans.

        A recent cross-cultural study by Danish and Chinese researchers supports the idea that culture influences taste interactions. Danish consumers experienced less sourness suppression by sucrose's sweetness compared to Chinese consumers. However, the reverse was not proven true, indicating that differences in taste perception may require further classification at the individual level.

        Sucrose (table sugar) undergoes hydrolysis, breaking down into glucose and fructose. The presence of acids, such as citric acid in fruit juices, accelerates this process, reducing sugar crystal formation and explaining how sourness suppresses sweetness. Conversely, when citric acid is introduced to sucrose, the resulting solution's pH range demonstrates how sweetness suppresses sourness.

        A separate Canadian study in 2019 found evidence linking certain taste sensitivities with taste preferences, with consumers who recorded higher sensitivities to sweetness and saltiness preferring these tastes less.

        The Danish and Chinese study investigated taste interactions in six liquid mixtures. Participants rated the taste sensitivity of each sample on a 9-point scale and evaluated mixtures containing sweetness and sourness on a 'Just About Right' (JAR) scale. The researchers found Danish consumers consistently recorded higher sweetness sensitivities than Chinese participants.

        However, the researchers could not conclude whether culture influenced sweetness suppression by sourness, as individual results varied regardless of nationality. Test subjects were later divided into three clusters based on their sourness sensitivity, and similar classification was performed for sweetness.

        Although certain trends suggested a relationship between culture and taste interactions, the researchers could not confirm this conclusion. Beverage manufacturers can use these findings to re-evaluate product segmentation strategies and diversify their target consumer range by focusing on factors such as age and gender rather than culture.

        In conclusion, this research challenges stereotypes regarding taste perceptions and preferences across cultures. It remains to be seen how people from different backgrounds will appreciate unique drinks like sugarcane juice with lemon in the future.
        """)
    elif select == "Text 2c: News Article 2 - Optimised (Max)":
        st.subheader("Investigating the Relationship Between Culture and Sweet-Sour Taste Interactions")
        st.write("*Examining Cultural Stereotypes in Taste Perceptions and Preferences*")
        st.write("By: ChatGPT (26 April 2023)")
        st.markdown("""
        Picture yourself sipping a margarita. After that first taste, you find the drink too sour. You lick some salt from the rim and take another sip. Now, the margarita tastes less sour. This is a prime example of a taste interaction between different taste qualities. When two or more taste qualities interact, they can either enhance or suppress one another, depending on their concentrations. The question arises: do cultural differences influence these taste interactions and perceptions?

        Prior studies have demonstrated that cultural differences impact taste sensitivities and interactions among individuals. A US study found that taste sensitivities across all five taste qualities were lower among African-American and Hispanic individuals compared to Caucasians. Another investigation revealed that Taiwanese students had higher sweetness sensitivity compared to their American counterparts.

        A recent study by Danish and Chinese researchers sought to validate and expand upon these findings. The cross-cultural study aimed to determine if culture affects taste interactions, particularly sweetness and sourness. Results indicated that Danish consumers experienced less sourness suppression by the sweetness of sucrose compared to their Chinese counterparts.

        Interestingly, researchers found no relationship between culture and the extent of sourness suppressing sweetness. This suggests that differences in taste perception might need further classification at the individual level.

        To comprehend the suppression between sweetness and sourness, it's essential to understand the chemical processes involved. Sucrose, or table sugar, undergoes hydrolysis, breaking down into glucose and fructose. Introducing acids like citric acid, commonly found in fruit juices, accelerates this process. As hydrolysis reduces sugar crystal formation, it explains how sourness can suppress sweetness.

        Similarly, citric acid's low pH of 2.2 indicates its high acidity and sourness. In contrast, sucrose is slightly alkaline and mildly bitter due to its higher pH of 8. When citric acid is introduced to sucrose, the resulting solution has a pH between 4 and 7, illustrating how sweetness can suppress sourness.

        A separate study conducted by Canadian researchers in 2019 found a relationship between certain taste sensitivities and preferences. Sweetness and saltiness were less preferred by individuals with higher sensitivities in these taste qualities.

        The study's participants included both Danish and Chinese individuals who evaluated six liquid mixtures: water, sucrose, tartaric acid, citric acid, sucrose with tartaric acid, and sucrose with citric acid. Tartaric and citric acids represented sourness, while sucrose exemplified sweetness.

        Participants tasted one sample at a time, rinsing their mouths with water and waiting 30 seconds between samples. Each sample's taste sensitivity was rated on a 9-point scale, ranging from 'not at all' (1) to 'extremely' sweet or sour (9), depending on the sample.

        For mixtures containing sweetness and sourness, researchers used an additional 'Just About Right' (JAR) scale to measure each individual's preferred sucrose and acid concentrations. The JAR ratings and taste sensitivities were recorded using a questionnaire for later analysis.

        On average, Danish consumers consistently reported higher sweetness sensitivities than their Chinese counterparts. This result supports the conclusion that sucrose suppressed tartaric acid to a greater extent in Chinese consumers compared to Danish consumers.

        The researchers acknowledged that a Caucasian population generally tends to have lower taste sensitivity for sweetness, sourness, saltiness, and bitterness than an Asian population. However, the inverse relationship between taste sensitivity and preference, established in the Canadian study, means that comparing taste sensitivities between Chinese and Danish consumers may not represent a complete 'East vs. West' comparison.

        Additionally, the researchers could not conclusively determine whether culture played a role in sweetness suppression by sourness. The results varied between individuals, regardless of their cultural backgrounds.

        Participants were grouped into clusters based on their relative sensitivities to sourness, with each cluster containing both Chinese and Danish consumers. While some trends suggested a relationship between culture and taste interactions, researchers could not confirm this conclusion upon further analysis.

        Beverage manufacturers can benefit from this study by re-evaluating their product segmentation strategies to diversify their target consumer range. Instead of focusing on culture, companies may want to explore other variables such as age and gender.

        With cultural stereotypes surrounding taste perception debunked, will people from different cultures appreciate unique drinks, like sugarcane juice with lemon, in the same way? Only time and experience will tell. Further research is needed to understand individual differences in taste perception and interactions, as well as the factors contributing to these differences.

        The study on sweet-sour taste interactions has shed light on the complex relationship between culture and taste perception. While it found that culture does influence taste interactions to some extent, it also revealed that individual differences play a crucial role. This research contributes valuable information to the understanding of taste perception and preferences and has practical applications in the beverage industry. However, more studies are needed to explore the nuances of taste perception further and to better understand the factors that contribute to individual differences in taste experiences.
        """)
        
      
elif selected == "Detailed Walkthrough":
    st.subheader("Detailed Walkthrough")
    check = ["Preliminary analysis", "Word clouds", "Identifying top 10 words within each article series"]
    confirm = st.selectbox("Which part of the detailed walkthrough are you reading?", options = check)
    st.write("Current selection:", confirm)
    if confirm == "Preliminary analysis":
        st.subheader("Preliminary analysis")
        st.write("After loading the article variants as text files and the necessary packages from NLTK, we can then analyse these texts before summarising their respective scores in the dataframe below:")
        st.write(backend.df)
        st.write("Comparing word counts of all 6 article variants")
        fig_a = backend.create_word_count_plot(backend.df)
        st.pyplot(fig_a)
        st.write("Comparing readability scores of all 6 article variants")
        fig_b = backend.create_flesch_reading_ease_plot(backend.df)
        st.pyplot(fig_b)
        st.write("Comparing sentiment compound scores of all 6 article variants")
        fig_c = backend.create_sentiment_compound_plot(backend.df)
        st.pyplot(fig_c)
    elif confirm == "Word clouds":
        st.subheader("Word clouds")
        texts = [backend.text1a, backend.text1b, backend.text1c, backend.text2a, backend.text2b, backend.text2c]
        text_ids = ["text1a", "text1b", "text1c", "text2a", "text2b", "text2c"]

        for text, text_id in zip(texts, text_ids):
            fig = backend.generate_word_cloud(text, text_id)
            st.pyplot(fig)
    elif confirm == "Identifying top 10 words within each article series":
        st.subheader("Identifying top 10 words within each article series")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Display plots from backend.py
        #st.header("Top 10 Words in Article Variants of News Article 1")
        fig = backend.plot_top_proportions(backend.proportions_series_1, "Top 10 Words in Article Variants of News Article 1")
        st.pyplot(fig)
        #st.header("Top 10 Words in Article Variants of News Article 1")
        fig2 = backend.plot_top_proportions(backend.proportions_series_2, "Top 10 Words in Article Variants of News Article 2")
        st.pyplot(fig2)
        #st.header("Top 10 Common Words Across Variants of Both Articles")
        fig3 = backend.plot_top_proportions(backend.proportions_common, "Top 10 Common Words Across Variants of Both Articles")
        st.pyplot(fig3)

elif selected == "References":
    st.header("References")
    st.subheader("News Article 1 - Timing vaccination campaign to reduce measles infections")
    st.markdown("""
    1. Thakkar, N., Gilani, S. S. A., Hasan, Q., & McCarthy, K. A. (2019). Decreasing measles burden by optimizing campaign timing. Proceedings of the National Academy of Sciences, 201818433. Also available from https://www.pnas.org/content/pnas/116/22/11069.full.pdf 
    2. Patel, M. K., Dumolard, L., Nedelec, Y., Sodha, S. V., Steulet, C., Gacic-Dobo, M., ... & Goodson, J. L. (2019). Progress toward regional measles elimination—worldwide, 2000–2018. Morbidity and Mortality Weekly Report, 68(48), 1105. Also available from https://www.cdc.gov/mmwr/volumes/68/wr/pdfs/mm6848a1-H.pdf 
    3. National Institute of Population Studies (NIPS) [Pakistan] and ICF. 2019. Pakistan Demographic and Health Survey 2017-18. Islamabad, Pakistan, and Rockville, Maryland, USA: NIPS and ICF. Also available from https://dhsprogram.com/pubs/pdf/FR354/FR354.pdf 
    4. Pakistan Bureau of Statistics. Block Wise Provisional Summary Results of 6th Population & Housing Census-2017 [As on January 03, 2018]. Also available from http://www.pbs.gov.pk/content/block-wise-provisional-summary-results-6th-population-housing-census-2017-january-03-2018 
    5. Centers for Disease Control and Prevention. Epidemiology and Prevention of Vaccine-Preventable Diseases. Chapter 10, Measles. 8th Edition, 2004. https://www.cdc.gov/vaccines/pubs/pinkbook/meas.html 
    """)
    st.subheader("News Article 2 - Investigating the relationship between culture and sweet-sour taste interactions")
    st.markdown("""
    1. Bertino, M., Beauchamp, G. K., & Jen, K. L. C. (1983). Rated taste perception in two cultural groups. Chemical senses, 8(1), 3-15. Also available from https://academic.oup.com/chemse/article-abstract/8/1/3/271785 
    2. Chamoun, E., Liu, A. A., Duizer, L. M., Darlington, G., Duncan, A. M., Haines, J., & Ma, D. W. (2019). Taste sensitivity and taste preference measures are correlated in healthy young adults. Chemical senses, 44(2), 129-134. Also available from https://pubmed.ncbi.nlm.nih.gov/30590512/ 
    3. Hydrolysis. (n.d). In Wikipedia. Retrieved from https://en.wikipedia.org/wiki/Hydrolysis 
    4. Junge, J.Y.; Bertelsen, A.S.; Mielby, L.A.; Zeng, Y.; Sun, Y.-X.; Byrne, D.V.; Kidmose, U. Taste Interactions between Sweetness of Sucrose and Sourness of Citric and Tartaric Acid among Chinese and Danish Consumers. Foods 2020, 9, 1425. Also available from https://www.mdpi.com/2304-8158/9/10/1425 
    5. Williams, J. A., Bartoshuk, L. M., Fillingim, R. B., & Dotson, C. D. (2016). Exploring ethnic differences in taste perception. Chemical senses, 41(5), 449-456. Also available from https://academic.oup.com/chemse/article/41/5/449/2366044 
    """)