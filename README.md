#### INTRODUCTION
In the previous report, it was suggested that Advance Queensland has been doing a good job in allocating funding to different sectors in Queensland to boost innovation, and commercialisation, and introduce new sectors and jobs to the market. We also know that the government has been adjusting the programs and requirements to effectively target each cohort. Since the Aboriginal and Torres Strait Islanders is one of the priority cohorts, it is necessary to look into the delivery of these programs to this group. 

As Queensland is heading to hosting the Olympic and Paralympic Games in 2032, innovation is the most important thing ever. Advance Queensland is one of the government initiatives to ensure better services and infrastructures to meet the needs of the estimated large amount of tourists visiting Queensland in 2032. In the past, [Indigenous languages had been showcased in the Women’s World Cup](https://www.brisbanetimes.com.au/national/queensland/meanjin-messaging-indigenous-languages-set-to-be-spoken-at-olympic-venues-20231110-p5ej48.html) and created a great effect to help inspire and develop the value of culture and understanding of culture and shared understanding. The Brisbane 2032 Olympic games are a great opportunity to develop a better cultural understanding of the First Nation people, not only by speaking the languages but also to foster collaboration to bring the Aboriginal culture closer to everyone. The effect of the Brisbane Olympic Games increase [the tourist expenditure revenue by $12 billion](https://www.dtis.qld.gov.au/__data/assets/pdf_file/0004/1626448/towards-2032-reshaping-queenslands-visitor-economy.pdf), making Queensland Australia’s destination of choice for domestic and global visitors seeking the world’s best experiences. Therefore, it is the goal of the government to grow visitation to Indigenous product and experiences, which is the concept of Indigenous tourism.  

This report will look into the existing Advance Queensland funding programs that support the Aboriginal and Torres Strait Islander group and also discover other factors that the government can focus on to improve the quality of the programs in the future. In order to do that, I used a combination of structured data from the Advance Queensland dataset and unstructured data from Guardian API.  

It is also important to identify appropriate terminology for our target cohort. According to the [Australian Institute of Aboriginal and Torres Strait Islander Studies](https://aiatsis.gov.au/explore/indigenous-australians-aboriginal-and-torres-strait-islander-people), there are more than 370 million Indigenous peoples spread across 70 countries worldwide, each practising unique traditions, retaining social, cultural, economic and political characteristics that are distinct from those of the dominant societies in which they live. The term ‘Indigenous Australian’ is used to encompass both Aboriginal people and Torres Strait Islander people and is used alternately in this report. Moreover, the target of Advance Queensland is not every Aboriginal and Torres Strait Islander people, but rather to focus on <b>Aboriginal and Torres Strait Islanders</b> who are <b>business owners, innovators, researchers and students </b>. This is an important stakeholder of Advance Queensland and the focus of this report.

#### 1. SCOPE OF FOCUS
In respect of the current tourism landscape that I mentioned, and the goals of Advance Queensland to bring innovation to their priority cohort, the most important question and its follow-up questions are:  

<b>How should the Queensland government design better programs for the Aboriginal and Torres Strait Islander cohort?</b>
- What do the current funds for Aboriginal and Torres Strait Islanders look like?
- What are the aspects of the Aboriginal being discussed?
- Are there any commercialisation opportunities for Aboriginal art and culture?

These are important questions to answer because [without better preversing their art and culture, the Indigenous traditional knowledge and language will slowly disappear.](https://press.un.org/en/2019/hr5431.doc.htm). This requires careful planning and preparation to effectively protect and transmit this knowledge from one generation to the other. While the Olympic 2032 is bringing a large amount of tourists around the country and the world to Brisbane, it is likely that visitors will visit many destinations in the city and also seek more original experiences outside Brisbane. If the governments are able to support this cohort and commercialise the rich cultural heritage of Australia’s Indigenous peoples, it is a good opportunity to enhance its reach to the international market.

#### DATA
Data source: Guardian API. This unstructured data can be obtained via an API key. 
Library and packages: nltk, json, string, sklearn for feature extraction.
A total of 100 articles were successfully fetched from the Guardian server.
![image](https://github.com/user-attachments/assets/f3ed1d34-ec1c-4a1f-a655-8fba230d54b1)

#### ANALYSIS AND VISUALISATION
### What does the funding for Aboriginal and Torres Strait Islanders look like?
There are 5 programs for Aboriginal and Torres Strait Islander groups recorded in this dataset. 
The amount of funding contributed to these programs is $2,265,695.0.
Compared to the total amount of all programs, these 10 programs accounted for 1.06% of the total funding.
### What are the most discussed aspects of Aboriginal and torres strait islander that the government can focus on?
![image](https://github.com/user-attachments/assets/977c6172-6e18-45d7-b494-886dc64fa2b1)
I used WordCloud to show the frequency of each term, where it can easily be seen that some words are significantly more common than another. <b>People</b>, <b>government</b>, <b>woman</b> are the top 3 most frequent words in these articles.
- In the articles, the word <b>people</b> is used when refering to Aboriginal and Torres Strait Islander as a whole, and also different "tribes" within this group, such as Gooniyandi people, Tjupan people, Dja Wurrung people, etc. It highlights that while the community can be referred as "Indigeous Australians" as a whole, this is a multicultural community that every idividual groups should be respected and recognised.
- As mentioned above, frequency of the word <b>government</b> indicates the significant role of governmental and political views in the Indigeous subject, especially in closing the cultural gap and build trust between communities and governments, as reported by the [Department of Prime Minister and Cabinet](https://www.niaa.gov.au/sites/default/files/reports/closing-the-gap-2017/executive-summary.html). Once again, this reflects the necessity of understanding and improving the impact of Advance Queensland programs to this cohort.
- The word <b>woman</b> shows the aspects of woman in the Indigenous community. A majority of it is about woman's health organisation, which suggested that the people starting to looking to balance of genders and their rights among the Aborignal and Torres Straite Islander community.

###  Is there any commercialisation opportunities for Aboriginal art and culture?
![image](https://github.com/user-attachments/assets/8c0f5325-0ba8-469f-8a36-b8705bb438d5)
To represent these features in respect to each article, I converted the count_dt_matrix using toarray(). Similar to the feature_names, the count_dt_matrix does not show article names. Therefore, I use the keys of my_articles, which is the name of each article at the beginning as the index of this dataframe.
![image](https://github.com/user-attachments/assets/89584e32-6732-4412-8cd1-8034f9c3523b)
As mentioned, I wanted to use LDA for topic modelling. I created a terms_df with 2 placeholders: count and lda to later add the results of these techniques for comparison.
![image](https://github.com/user-attachments/assets/b30f330d-1b36-4944-a523-1d67cc0430fe)
In order to add the top terms of each article, I ran the code that looped through all the articles, and identified its row in the `count_df` and `sort_values` of each feature in each document by the number it appears in the document. Then I used `head(10)` to show the most significant words. Then the code will list these words in the `count` column according to the index of the article.

While both LDA (Latent Dirichlet Allocation) and NMF (Non-negative Matrix Factorization) are common topic modelling techniques, especially when prior knowledge of the topic is unknown, I chose LDA to perform topic modelling because it was suggested that LDA can perform better in capturing complex relationships between topics ([Topic modeling algorithms](https://medium.com/@m.nath/topic-modeling-algorithms-b7f97cec6005)).  
I set the num_topic at 6. Since the topics are not defined, setting the number of topics too high can reduce differences among topics. On the other hand, setting a low number of topics can result in one topic carrying a mix of different topics because the number of clusters does not allow articles that have quite different topics from each other to be separated into different groups.

![image](https://github.com/user-attachments/assets/30f42cee-24ab-48cb-907b-ce7fa070a3fd)


![image](https://github.com/user-attachments/assets/ca91008a-353a-488e-978c-eed5431b5b41)
Although LDA outputs a set of distinct topics, it is important to look into the number of articles under each topic. For each article in the doc_topic_matrix, `argmax()` will allocate the topic with the highest score of the article and to count the number of articles belong to the topic.
Since the topic and its order may differ every time the function is run, I sorted them by the number of article they contains to locate and investigate those articles.

![image](https://github.com/user-attachments/assets/95f22203-3940-48f1-8d84-13cc8c79f6d8)
We can view the articles under each topics by again using argmax() to get the most relevant topic of each article. For example, I am interested to see the result of the most popular topic, I will refer to the doc_per_topic_df which is already sorted by the number of articles it contains. Some articles indicated that some popular Indigenous culture and art experiences are about tour, food, community. There are some specific experienced that people talked about, such as tree-hugging and camping. Other articles suggested the leasure activities that people are interested in, such as portrait painting, sound healing, going for music shows, etc. These insights gave some very good recommendation to bringing Indigenous cultures to meet the exploration needs of people nowadays and make them a commercialise opportunity while still preserving the integrity of this ancient art form.

#### INSIGHTS
The purpose of this report is to further address some important questions about the Queensland funding situation, specifically to the Aboriginal and Torres Strait Islander community.  

1. What do the current funds for Aboriginal and Torres Strait Islanders look like?
There are a few programs that are indicated to be for the Aboriginal and Torres Strait Islander, including Aboriginal and Torres Strait Islander PhD Scholarships, Aboriginal and Torres Strait Islander Research Fellowships, Deadly Data, Deadly Innovation, Yarrabah Business Accelerator Incubation Hub. The amount of funds offered to this cohort is significantly lower than other groups due to the small number of Indigenous businesses and start-ups in the market. Some programs have been significantly boosting the commercialization of Indigenous culture and bringing about a more stable income source for many Aboriginal artists and their families, improving their living conditions and allowing them to pursue their craft full-time.
2. What are the aspects of the Aboriginal being discussed?
People, government and women are the most discussed phrases when searching for the Aboriginal and Torres Strait Islander topic. These phrases indicate what aspects are important to the Indigenous community and the necessity of the government recognising different local Indigenous communities, rather than treating them as a whole. 
3. Are there any commercialisation opportunities for Indigenous art and culture?
From the topic modelling, it was identified that some special Indigenous traditions can be interesting to visitors, such as tree-hugging traditions, art, food, and festival experiences. The topic modelling also indicated the need for healing activities with art, sounds, light, etc. Further research into these topics can be beneficial for Queensland to create an amazing experience for tourists coming to Brisbane to enjoy the Olympic games and explore the uniqueness of the Aboriginal and Torres Strait Islander history and tradition.

<b>How should the Queensland government design better programs for the Aboriginal and Torres Strait Islander cohort?</b>  
From the analysis of the Advance Queensland dataset and further exploring unstructured data from Guardian API, it is suggested that the government can turn their focus on individual ethnic groups within the Aboriginal and Torres Strait Islander community to identify different cultures and traditions that can be turned into a tourism opportunity. Further research into the needs of tourists nowadays can be beneficial to the planning and preparation in order to ensure the best experiences for travellers to Brisbane, especially in 2032, which is the peak economic opportunity for Queensland.

#### LIMITATIONS
- First of all, the Advance Queensland dataset provides limited information about the recipients' ethnicity. Therefore, the analysis only captured the programs that were designed for the Indigenous cohort without taking into account invidudual projects run by Indigenous people or bring benefits to the Indigenous community.
- Secondly, the LDA model is not sufficient to indicate the public's opinion on the Indigenous subject, while this information can be beneficial to decision-makers. 
