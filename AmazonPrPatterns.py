""" Modules for scraping and processing Amazon Press Release Information"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import warnings
from datetime import datetime
import string
import spacy
import en_core_web_sm
from fbprophet import Prophet
from sklearn.feature_extraction import text
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from fbprophet import Prophet
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


class PrReScrape:
    def __init__(self, startyear, endyear, content_grab = 'false'):
        years = list(range(startyear, endyear+1))
        amazon_urls = []
        for year in years:
            amazon_url = f'https://press.aboutamazon.com/press-releases?a9d908dd_year%5Bvalue%5D={year}&op=Year+Filter&a9d908dd_widget_id=a9d908dd&form_build_id=form-VCLgnpphxz4pwMHMkBZPwf2vTuTX5iywluCMBYpryd8&form_id=widget_form_base'
            amazon_urls.append(amazon_url)
        self.amazonurls = amazon_urls
        warnings.warn('URLs compiled. Operation could take between 30 and 60 minutes to complete scrape')
        press_release_dates = []
        headlines = []
        headline_url = []
        for url in self.amazonurls:
            response = requests.get(url)
            page = response.text
            soup = BeautifulSoup(page, 'html5lib')
            date = soup.find('div', class_='nir-widget--field nir-widget--news--date-time')
            press_release_dates.append(date.text.strip())
            release_url = soup.find('div', class_='nir-widget--field nir-widget--news--headline')
            target_url = release_url.find('a')['href']
            headline = release_url.text.strip()
            headline_url.append(target_url)
            headlines.append(headline)
            while date.findNext('div', class_="nir-widget--field nir-widget--news--date-time")!=None:
                date = date.findNext('div', class_="nir-widget--field nir-widget--news--date-time")
                press_release_dates.append(date.text.strip())
                release_url = release_url.findNext('div', class_='nir-widget--field nir-widget--news--headline')
                target_url = release_url.find('a')['href']
                headline = release_url.text.strip()
                headline_url.append(target_url)
                headlines.append(headline)
        self.headlines = headlines
        self.headline_url = headline_url
        print('Headlines Compiled')
        contents = []
        dates = []
        times = []
        count=1
        for tail in headline_url:
            base = 'https://press.aboutamazon.com/'
            url = base+tail
            response = requests.get(url)
            page = response.text
            soup = BeautifulSoup(page, 'html5lib')
            if content_grab=='true':
                try:
                    content = soup.find('div', class_='node__content').text.strip().replace('\n', ' ').replace('\xa0', ' ')
                except:
                    content = 0
                try:
                    content = content.split('-- ')[-1]
                except:
                    pass
                try:
                    content = content.split('About Amazon.com')[0]
                except:
                    pass
                contents.append(content)    
            try:
                date_time = soup.find('div', class_ = 'field field--name-field-nir-news-date field--type-datetimezone field--label-hidden').text.replace('\n', '').strip().split(' at ')
                date = date_time[0]
                time = date_time[1].replace(' EST', '')
            except:
                date = tail
                time = tail
            dates.append(date)
            times.append(time)
            if count%250==0:
                print(f'{count} press releases have been processed. Processing ongoing...')
            count+=1
        self.dates = dates
        self.times = times
        self.contents = contents
        df_content = pd.DataFrame([headline_url, dates, times, contents]).T
        df_content.columns = ['link_suffix', 'date', 'time', 'content']
        df_content['hour'] = time.split(':')[0].strip()
        df_content['minute'] = time.split(':')[1].split(' ')[0].strip()
        df_content['time_of_day'] = time.split(':')[1].split(' ')[1].strip()
        df_content['hour'] = df_content['hour'].astype(float)
        df_content['minute'] = df_content['minute'].astype(float) / 60
        df_content['am/pm'] = 0
        df_content.loc[df_content['time_of_day']=='PM', 'am/pm']=12
        df_content['time_adj'] = df_content['hour']+df_content['am/pm']+df_content['minute'] 
        df_content.drop(['hour', 'minute', 'time_of_day', 'am/pm'], axis = 1, inplace = True)
        print('Press release content and release time compiled')
        df_amazon = pd.DataFrame([press_release_dates, headlines, headline_url]).T
        df_amazon.columns = ['date', 'headline', 'link']
        df_amazon['date'] = pd.to_datetime(df_amazon['date'])
        df_amazon = pd.merge(df_amazon, df_content, how = 'inner', left_on='link', right_on='link_suffix')
        df_amazon.drop(['link_suffix', 'date_y'], axis=1, inplace=True)
        df_amazon.rename(columns = {'date_x': 'date'}, inplace=True)
        self.dataframe = df_amazon
        warnings.warn('Dataframe compiled. Datframe can be accesses by calling "PrReScrape.dataframe" or by calling "PrReScrape.getdata()".')
        
    def getdata(self):
        return self.dataframe
    
    def savedata(filepath):
        return self.dataframe.to_csv(filepath)
    
    def textprocess(self, savename, ContentOrHeadline = 'headline'):
        if ContentOrHeadline == 'content':
            warnings.warn('Although partial press release content cleaning is available, further analysis is required to propertly clean since spacing issues occur at the web scrape stage. As a result, headline analysis is highly recommended. To complete the project end-to-end with content analysis, further code must be written to fix web scrape issues.')
            alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
            quotations_one = lambda x: re.sub('“', ' ', x)
            quotations_two = lambda x: re.sub('”', ' ', x)
            punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower()) 
            df_amazon = self.dataframe
            df_amazon['content_text'] = df_amazon.content.apply(str).map(alphanumeric).map(punc_lower).map(quotations_one).map(quotations_two)
            
            nlp = en_core_web_sm.load()
            
            lemma_list = []
            for headline in df_amazon['content_text'].apply(str):
                lemma = []
                nlp_comment = nlp(headline)
                for word in nlp_comment:
                    lemma.append(word.lemma_)
                lemma_list.append(lemma)
            df_amazon['lemma_content'] = lemma_list
            
            pos_list = []
            for comment in df_amazon['content_text'].apply(str):
                pos = []
                nlp_comment = nlp(comment)
                for word in nlp_comment:
                    pos.append(word.pos_)
                pos_list.append(pos)
            df_amazon['pos_content'] = pos_list
            
            lab_list = []
            for comment in df_amazon['content_text'].apply(str):
                label = []
                nlp_comment = nlp(comment)
                for word in nlp_comment.ents:
                    label.append(word.label_)
                lab_list.append(label)
            df_amazon['label_content'] = lab_list
            
            cleaned_list = []
            for i in range(len(df_amazon.lemma_content)):
                comment = list(zip(df_amazon.lemma_content[i], df_amazon.pos_content[i]))
                words = []
                for row in comment:
                    if row[1]=='NOUN' or row[1]=='ADJ' or row[1]=='VERB' or row[1]=='PROPN':
                        words.append(row[0])
                cleaned_list.append(words)
            df_amazon['cleaned'] = cleaned_list
            
        if ContentOrHeadline == 'headline':
            alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
            quotations_one = lambda x: re.sub('“', ' ', x)
            quotations_two = lambda x: re.sub('”', ' ', x)
            punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
            df_amazon = self.dataframe
            df_amazon['headline_text'] = df_amazon.headline.apply(str).map(alphanumeric).map(punc_lower).map(quotations_one).map(quotations_two)
            
            nlp = en_core_web_sm.load()
            
            lemma_list = []
            for headline in df_amazon['headline_text'].apply(str):
                lemma = []
                nlp_comment = nlp(headline)
                for word in nlp_comment:
                    lemma.append(word.lemma_)
                lemma_list.append(lemma)
            df_amazon['lemma_headline'] = lemma_list
            
            pos_list = []
            for comment in df_amazon['headline_text'].apply(str):
                pos = []
                nlp_comment = nlp(comment)
                for word in nlp_comment:
                    pos.append(word.pos_)
                pos_list.append(pos)
            df_amazon['pos_headline'] = pos_list
            
            lab_list = []
            for comment in df_amazon['headline_text'].apply(str):
                label = []
                nlp_comment = nlp(comment)
                for word in nlp_comment.ents:
                    label.append(word.label_)
                lab_list.append(label)
            df_amazon['label_headline'] = lab_list
            
            cleaned_list = []
            for i in range(len(df_amazon.lemma_headline)):
                comment = list(zip(df_amazon.lemma_headline[i], df_amazon.pos_headline[i]))
                words = []
                for row in comment:
                    if row[1]=='NOUN' or row[1]=='ADJ' or row[1]=='VERB' or row[1]=='PROPN':
                        words.append(row[0])
                cleaned_list.append(words)
            df_amazon['cleaned'] = cleaned_list
        self.dataframe = df_amazon
        self.dataframe.to_csv(f'{savename}.csv')
        return df_amazon

def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
    
class LdaOutperformance:
    '''
    takes in: (ticker, startdate, enddate, csv_savename, stopwords (in list format), starting_number_of_LDA_groupings, ending_number_of_LDA_groupings, how to measure outperformance (threshold or quartile measurement, measurement stat))
    Ticker = as listed on finance.yahoo.com
    startdate & enddate = yyyy-mm-dd
    '''
    
    def __init__(self, ticker, csv_savename, start_group, end_group, scrape_dataframe, startdate = '2012-01-01', enddate = None, outperformance = ('threshold', 0), effective=.65, stop_words = [] ):
        
        # Download the target company's daily stock price over the period from finance.yahoo.com
        df_amazon = pd.read_csv(f'{scrape_dataframe}.csv')
        if startdate!=None:
            df_amazon = df_amazon[df_amazon['date']>=startdate].copy()
        if enddate!=None:
            df_amazon = df_amazon[df_amazon['date']<=enddate].copy()
        max_date = max(pd.to_datetime(df_amazon['date']))
        self.max_date = max_date
        min_date = min(pd.to_datetime(df_amazon['date']))
        self.min_date = min_date
        start_year = int(min_date.year)
        start_month = int(min_date.month)
        start_day = int(min_date.day)
        end_year = int(max_date.year)
        end_month = int(max_date.month)
        end_day = int(max_date.day)
        start_timestamp = datetime(min_date.year, min_date.month, min_date.day) - datetime(1970, 1, 1)
        start = int(start_timestamp.total_seconds())
        end_timestamp = datetime(max_date.year, max_date.month, max_date.day) - datetime(1970, 1, 1)
        end = int(end_timestamp.total_seconds())
        url = f'https://query1.finance.yahoo.com/v7/finance/download/AMZN?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true'
        req = requests.get(url)
        url_content = req.content
        csv_file = open(f'{csv_savename}.csv', 'wb')
        csv_file.write(url_content)
        csv_file.close()
        target_pricing = pd.read_csv(f'{csv_savename}.csv')
        target_pricing['Date'] = pd.to_datetime(target_pricing['Date'])
        if startdate!=None:
            target_pricing = target_pricing[target_pricing['Date']>=startdate].copy()
        if enddate!=None:
            target_pricing = target_pricing[target_pricing['Date']<=enddate].copy()
            
        self.target_pricing = target_pricing
        
        # Set up LDA analysis environment for booklet of retrieved press releases
        my_stop_words = text.ENGLISH_STOP_WORDS.union(['amazonlocalamazonlocal','seasonamazonsmile', 'refrigeratorsave', 'workforcethe', 'bankstandard', 'moviesincludinganchorman', 'softwaretoyssave', 'amazonsmartoven', 'counterfeitersamazon', 'chargersspotlighte', 'collectionsave', 'audiblewith', 'wordsreleasing', 'kaytranadakelsey', 'backyardigan', 'kindletouchintl', 'gameconnect', 'filamentadditional', 'vacuumsave', 'applicationswhispercast', 'cardsave', 'marketcart', 'hallkristen', 'followingpbs', 'nerfsave', 'camperforce', 'collectionsave', 'unionagent', 'contentin', 'windowsphoneapp','primevideolanding','orgeousbelize','contentin', 'kaytranadakelsey','freemayday','largersave', 'startupchallenge', 'windowprovide', 'storeand', 'anyconnect', 'capgemini', 'nesteremail', 'excitedto', 'ipadalso','countdownpowere', 'pharmaceuticalcompanieslike', 'cardssave', 'kindleupdate', 'appseattle', 'parentswith','relationsdonna','excitedto','amazonfor','com', 'introduce', 'amazon', 'nasdaq', 'amazonamazonis', 'byamazon', 'visitguardianlife', 'nysheets', 'workatamazonfulfillment'])
        my_stop_words = my_stop_words.union(stop_words)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        tfidf_vectorizer = TfidfVectorizer(strip_accents = 'unicode',
                                        stop_words = my_stop_words,
                                        lowercase = True,
                                        token_pattern = r'\b[a-zA-Z]{3,}\b',
                                        ngram_range = (1,1),
                                        max_df = 0.1, 
                                        min_df = 1)
        self.tfidf_vectorizer = tfidf_vectorizer
        dtm_tfidf = tfidf_vectorizer.fit_transform(df_amazon.cleaned.apply(str))
        self.dtm_tfidf = dtm_tfidf
        
        # Create dataframe outlining topic assignment for a given number of topic groupings - will be used to find the most efficient number of topics
        chart = []
        topic_number = []
        for num in range(start_group, end_group+1):
            lda_tfidf = LatentDirichletAllocation(n_components=num , random_state=0)
            lda_tfidf.fit(dtm_tfidf)
            outcome = lda_tfidf.transform(dtm_tfidf)
            topic_score = pd.DataFrame(outcome)
            topics = topic_score.idxmax(axis=1).to_list()
            chart.append(topics)
            topic_number.append(num)
        df_topics = pd.DataFrame(chart).T
        df_topics.columns = topic_number
        df_topics['date'] = df_amazon['date']
        df_topics['time'] = df_amazon['time_adj']
        self.df_topics = df_topics
        
        # Attach target pricing information to the topic assginment dataframe to understand daily price changes on date of each release
        amzn = target_pricing
        amzn['Date'] = pd.to_datetime(amzn['Date'])
        amzn['shift'] = amzn.Close.shift()
        amzn['daily_change'] = (amzn['Close']/amzn['shift'])-1
        self.amzn = amzn
        daily_change = amzn[['Date', 'daily_change']].copy()
        daily_change['daily_change_+1'] = daily_change['daily_change'].shift(-1)
        
        # Get benchmark info (Dow Jones Industrial) to establish reference for outperformance
        df_topics['date'] = pd.to_datetime(df_topics['date'])
        df1 = pd.merge(df_topics, daily_change, how ='inner', left_on = 'date', right_on = 'Date')
        dji_url = f'https://query1.finance.yahoo.com/v7/finance/download/AMZN?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true'
        req = requests.get(dji_url)
        url_content = req.content
        csv_file = open('dji_info.csv', 'wb')
        csv_file.write(url_content)
        csv_file.close()
        dji= pd.read_csv('dji_info.csv')
        dji['Date'] = pd.to_datetime(dji['Date'])
        if startdate!=None:
            dji = dji[dji['Date']>=startdate].copy()
        if enddate!=None:
            dji = dji[dji['Date']<=enddate].copy()
        dji['previous_day_close'] = dji.Close.shift()
        dji['daily_change'] = (dji['Close']/dji['previous_day_close'])-1
        dji['next_day_change'] = dji.daily_change.shift(-1)
        dji_moves = dji[['Date', 'daily_change', 'next_day_change']]
        dji_moves.columns = ['date', 'dji_daily_change', 'dji_next_day_change']
        df2 = pd.merge(df1, dji_moves, how='inner', left_on = 'Date', right_on = 'date')
        df2.drop(['date_x', 'date_y'], axis = 1, inplace=True)
        df2 = pd.merge(df1, dji_moves, how='inner', left_on = 'Date', right_on = 'date')
        df2.drop(['date_x', 'date_y'], axis = 1, inplace=True)
        amzn_moves = amzn[['Date', 'daily_change']]
        
        # Create a chart that determines target outperformance based on market expectations driven by Dow Jones Industrial
        beta_chart = pd.merge(dji_moves, amzn_moves, how = 'inner', left_on = 'date', right_on ='Date')
        beta_chart.drop('Date', axis = 1, inplace = True)
        beta_chart.rename(columns = {'daily_change':'amzn_daily_change'}, inplace = True)
        beta_chart_calc = beta_chart[['amzn_daily_change', 'dji_daily_change']]
        periods = int(252)  # Uses a one year beta calculation
        result = beta_chart_calc.dji_daily_change.rolling(window =periods).cov(beta_chart_calc.amzn_daily_change).to_list() / beta_chart.dji_daily_change.rolling(window = periods).var()
        result_df = pd.DataFrame(result).rename(columns = {'dji_daily_change':'amzn_beta_daily_oneyear'})
        df3 = pd.merge(df2, result_df, how = 'inner', left_index = True, right_index = True)
        df3['amzn_est_day_of'] = df3['dji_daily_change']*df3['amzn_beta_daily_oneyear']
        df3['amzn_est_next_day']=df3['dji_next_day_change']*df3['amzn_beta_daily_oneyear']
        df3['amzn_day_outperformance'] = df3['daily_change']-df3['amzn_est_day_of']
        df3['amzn_next_day_outperformance'] = df3['daily_change_+1']-df3['amzn_est_next_day']
        df3['amzn_outperform_associated'] = df3['amzn_day_outperformance']
        df3.loc[df3['time']<3.5, 'amzn_outperform_associated'] = df3['amzn_next_day_outperformance'] #If press release happens before 3:30pm EST measure outperformance against same day market movements. if after 3:30pm, use next day market movements as measurement.
        df3['outperform_true'] = 0
        if outperformance[0] == 'threshold':
            df3.loc[df3['amzn_day_outperformance']>=outperformance[1],'outperform_true'] = 1
        elif outperformance[0]== 'quartile':
            df3.loc[df3['amzn_day_outperformance']>=df3.amzn_day_outperformance.quantile(q=0.75),'outperform_true'] = 1
        df4=df3[df3['amzn_beta_daily_oneyear'].notna()]
        self.df4 = df4
        
        # Divide into two charts: upper_quart, lower_quart, with 1 representing if press release aligns with daily outperformance
        upper_quart_chart = df4[df4['outperform_true']==1].copy()
        lower_quart_chart = df4[df4['outperform_true']==0].copy()
        
        #create a dataframe showing effectiveness (defined as outperforming press releases/total press releases in one topic) for each topic in a given topic grouping
        topic_list = []
        for topic_nums in range (start_group+1,end_group):
            percent_list = []
            for topic in range(topic_nums):
                    upper_col = upper_quart_chart[topic_nums]
                    lower_col = lower_quart_chart[topic_nums]
                    upper_count = upper_col.value_counts()[topic]
                    lower_count = lower_col.value_counts()[topic]
                    percent = upper_count/(upper_count+lower_count)
                    percent_list.append(percent)
            topic_list.append(percent_list)
        topic_effectiveness = pd.DataFrame(topic_list).T
        self.topic_effectiveness = topic_effectiveness
        
        # Identifying the category with maximum effectiveness for each topic group
        effectiveness = []
        for col in range(topic_effectiveness.shape[1]):
            max_ = topic_effectiveness[col].max()
            effectiveness.append(max_)
        self.effectiveness = effectiveness
        self.max_measures = list(enumerate(effectiveness))
        self.highest_max_measure =(effectiveness.index(max(effectiveness)), max(effectiveness))
        
        # Identifying the topic group that captures the most number of outperforming press releases in high concentrations
        sums = []
        for topic_num in range(topic_effectiveness.shape[1]):
            counts = 0
            for topic in range(topic_num):
                if topic_effectiveness.iloc[topic, topic_num]>effective:
                    count = upper_quart_chart[topic_num].value_counts()[topic]
                else:
                    count = 0
                counts +=count
            sums.append(counts)
        self.effective = effective
        self.pr_outperform_capture = list(enumerate(sums))
        self.max_topic_grouping = (sums.index(max(sums))+1, max(sums))
        self.max_topic_grouping_id = sums.index(max(sums))+1
        
    def load_daily_prices(self):
        yahoo_csv = pd.read_csv(f'{self.csv_savename}.csv')
        return yahoo_csv
    
    def display_topics(self, feature_names = None, no_top_words = 15, topic_names=None):
        model = LatentDirichletAllocation(n_components = self.max_topic_grouping_id, random_state=0)
        model.fit(self.dtm_tfidf)
        outcome = model.transform(self.dtm_tfidf)
        if feature_names==None:
            feature_names=self.tfidf_vectorizer.get_feature_names()
        for ix, topic in enumerate(model.components_):
            if not topic_names or not topic_names[ix]:
                print("\nTopic ", ix)
            else:
                print("\nTopic: '",topic_names[ix],"'")
            print(", ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))

        
    def prophet_run(self, model = Prophet(), category = None, threshold = 0.65, quan = .5, rolling_measure=14, num_desc_words = 15, train_test_split = '2018-01-01'):
        
        # Establish objects created prior to prophet_run()
        
        topic_effectiveness = self.topic_effectiveness
        amzn = self.amzn
        df4 = self.df4
        dtm_tfidf = self.dtm_tfidf
        tfidf_vectorizer = self.tfidf_vectorizer
        
        
        # Initiate model
        
        model = model
        
        # Category (number of topics) defaults to max_topic_grouping_id, established during the __init__ function. Then creates a list "select_topics" which includes outperforming topics
        
        if category==None:
            category=self.max_topic_grouping_id
        select_topics = topic_effectiveness[topic_effectiveness[category]>threshold][category].index.to_list()
        print(select_topics)

        # Creates dataframe with isolated dates and catagories. Assigns an occurance value of one (to be appended later to the calendar)
        isolated_data = df4[df4[category].isin(select_topics)][['Date', category]].copy()
        isolated_data['occurance'] = 1
        occurance_calendar = pd.DataFrame(pd.Series(pd.date_range(str(self.min_date.year), freq='D', periods=(datetime(self.max_date.year, self.max_date.month, self.max_date.day) - datetime(self.min_date.year, 1, 1)).days)))
        occurance_calendar.columns = ['date']
        occurance_calendar = pd.merge(occurance_calendar, isolated_data, how='outer', left_on = 'date', right_on = 'Date')
        occurance_calendar.occurance.fillna(0, inplace=True)
        occurance_calendar['rolling'] = occurance_calendar.occurance.rolling(window=rolling_measure).sum()
#         y3 = occurance_calendar['rolling']
#         x3 = occurance_calendar['date']

        # Split into a training set and test set, seperated at the "train_test_split" date
        cleaned_calendar=occurance_calendar.copy().drop('Date', axis = 1)
        cleaned_calendar.sort_values('date', inplace = True)
        train = cleaned_calendar[cleaned_calendar['date']<train_test_split].copy()
        test = cleaned_calendar[cleaned_calendar['date']>=train_test_split].copy()

        
        # Conduct LDA analysis on the target grouping, print top 15 words for each topic (or change with num_desc_words)
        lda_tfidf= LatentDirichletAllocation(n_components=category, random_state=0)
        # lda_tfidf_content = LatentDirichletAllocation(n_components=15 , random_state=0)
        lda_tfidf.fit(dtm_tfidf)
        # lda_tfidf_content.fit(dtm_tfidf_content)
        display_topics(lda_tfidf, tfidf_vectorizer.get_feature_names(), num_desc_words)

        # format training dataframe for use in the fbprophet model. Create Future data to measure results
        model_train = train[['date', 'rolling']].copy()
        model_train.columns = ['ds', 'y']
        model.fit(model_train)
        future = model.make_future_dataframe(periods=(datetime(self.max_date.year, self.max_date.month, self.max_date.day) - datetime(int(train_test_split[:4]), int(train_test_split[5:7]), int(train_test_split[8:]))).days)
        
        # Predict on the test dates
        forecast = model.predict(future)
        pred = forecast[forecast['ds']>=train_test_split].copy()
        
        # Remove days of week where trading does not occur
        pred['day_of_week'] = pred.ds.dt.dayofweek
        pred = pred[(pred['day_of_week']!=5) & (pred['day_of_week']!=6)]
        pred = pred[pred.ds.isin(amzn['Date'].to_list())]
        
        #initiate column to indicate dates where prediction is considered likely. likely = 1, unlikely = 0. flex provided by "quant" parameter 
        pred['likely_to_occur'] = 0
        pred.loc[pred['yearly']>=pred.yearly.quantile(quan), 'likely_to_occur']=1
        pred['buy/sell_program'] = pred['likely_to_occur'].shift()
        pred['buy_sell'] = (pred['likely_to_occur']-pred['buy/sell_program'])*-1
        pred_append = pred[pred['buy_sell']!=0][['ds', 'buy_sell']].copy()
        pred_irr = pd.merge(pred_append, amzn[['Date', 'Open', 'Close']], how = 'left', left_on = 'ds', right_on = 'Date' )
        fig = model.plot(forecast)
        
        # add green line to graph to show trading days
        for point in pred_irr['ds'][1:]:
            plt.axvline(point,ls='--', lw=1, color='green')
        plt.show()
#         forecast_error(forecast, other_topic_df)
        print(pred_irr.shape)
#         pred_irr['cashflow'] = 0
#         pred_irr.loc[pred_irr['buy_sell']==-1, 'cashflow'] = pred_irr['buy_sell']*pred_irr['Open']
#         pred_irr.loc[pred_irr['buy_sell']==1, 'cashflow']=pred_irr['buy_sell']*pred_irr['Close']

        # Create chart showing where trading days occur in comparison to likelihood of topics occuring
        plot = forecast[forecast['ds']>=train_test_split][['ds', 'yhat']]
        plot = pd.merge(plot, amzn, how ='left', left_on = 'ds', right_on = 'Date' )
        x = plot['ds']
        y1 = plot['Close']
        y = plot['yhat']
        fig, ax=plt.subplots(figsize = (10,5))
        plt.plot(x, y)
        points=[]
        for point in pred_irr['ds'][1:]:
            plt.axvline(point,ls='--', lw=1, color='green')
            points.append(point)
        points.append(max(amzn['Date']))
        if len(points)%2!=0:
            points.append(pd.to_datetime(f'{self.max_date.year}-12-31'))
        buy_list = []
        sell_list = []
        for num in range(len(points)):
            if (num+1)%2!=0:
                buy = points[num]
                sell = points[num+1]
                plt.axvspan(buy, sell, facecolor="blue", alpha=0.25)
                buy_list.append(buy)
                sell_list.append(sell)
        self.buy_list = buy_list
        self.sell_list = sell_list
        pricing_days = amzn[['Date', 'Open', 'Close']].copy()
        trading_days_buy = pricing_days[pricing_days['Date'].isin(buy_list)].copy()
        trading_days_sell = pricing_days[pricing_days['Date'].isin(sell_list)].copy()
        trading_days_buy['trade'] = trading_days_buy['Open']*-1
        trading_days_sell['trade'] = trading_days_sell['Close']
        trades = trading_days_buy.append(trading_days_sell)
        trades = trades.sort_values(by='Date')
        self.trades = trades
        trades.to_csv(f'trades.csv')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Sans-Serif")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Sans-Serif")
        ax.set_facecolor('aliceblue')
        plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
        plt.tick_params(axis = "y", which = "both", left = False, right = False)
        years = YearLocator() 
        yearsFmt = DateFormatter('%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        plt.show;

        # Create chart showing where trading days occur in comparison to target company price
        fig, ax=plt.subplots(figsize = (10,5))
        plt.plot(x, y1)
        points=[]
        for point in pred_irr['ds'][1:]:
            plt.axvline(point,ls='--', lw=1, color='green')
            points.append(point)
        if len(points)%2!=0:
            points.append(pd.to_datetime(f'{self.max_date.year}-12-31'))
        for num in range(len(points)):
            if (num+1)%2!=0:
                plt.axvspan(points[num], points[num+1], facecolor="blue", alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Sans-Serif")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Sans-Serif")
        ax.set_facecolor('aliceblue')
        plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
        plt.tick_params(axis = "y", which = "both", left = False, right = False)
        years = YearLocator() 
        yearsFmt = DateFormatter('%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        plt.show;

        # Graph test prediction
        plot = forecast[forecast['ds']>=train_test_split][['ds', 'yhat']]
        x = plot['ds']
        y = plot['yhat']
        fig, ax=plt.subplots(figsize = (10,5))
        plt.plot(x, y)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Sans-Serif")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Sans-Serif")
        ax.set_facecolor('aliceblue')
        plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
        plt.tick_params(axis = "y", which = "both", left = False, right = False)
        Years = YearLocator() 
        yearsFmt = DateFormatter('%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        plt.show;

        # Graph training trend
        training = forecast[forecast['ds']<=train_test_split].copy()
        x = training['ds']
        y = training['yhat']
        fig, ax=plt.subplots(figsize = (10,5))
        plt.plot(x,y)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Sans-Serif")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Sans-Serif")
        ax.set_facecolor('aliceblue')
        plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
        plt.tick_params(axis = "y", which = "both", left = False, right = False)
        plt.show;

        # Graph both training and test set with line showing date division
        plot = forecast[['ds', 'yhat']]
        x = plot['ds']
        y = plot['yhat']
        fig, ax=plt.subplots(figsize = (10,5))
        ax.plot(x, y)
        ax.axvline(pd.to_datetime(train_test_split),ls='--', lw=4, color='green')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Sans-Serif")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Sans-Serif")
        ax.set_facecolor('aliceblue')
        plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
        plt.tick_params(axis = "y", which = "both", left = False, right = False)
        plt.show()

        # Graph date where press-releases occur against the prediction trend
        training = forecast[forecast['ds']<=train_test_split].copy()
        x = training['ds']
        y = training['yhat']
        fig, ax=plt.subplots(figsize = (10,5))
        plt.plot(x,y)
        occurance = occurance_calendar[(occurance_calendar['occurance']==1)&(occurance_calendar['date']<=train_test_split)]['date'].copy()
        for date in occurance:
            plt.axvline(date,ls='--', lw=1, color='green')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Sans-Serif")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Sans-Serif")
        ax.set_facecolor('aliceblue')
        plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
        plt.tick_params(axis = "y", which = "both", left = False, right = False)
        plt.show;

        # One Year Pattern
        training = forecast[(forecast['ds']>=f'{self.max_date.year}-01-01')&(forecast['ds']<f'{self.max_date.year}-12-31')].copy()
        x = training['ds']
        y = training['yhat']
        fig, ax=plt.subplots(figsize = (10,5))
        plt.plot(x,y)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Sans-Serif")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Sans-Serif")
        ax.set_facecolor('aliceblue')
        plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
        plt.tick_params(axis = "y", which = "both", left = False, right = False)
        months = MonthLocator() 
        monthsFmt = DateFormatter('%b')
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(monthsFmt)
        plt.show;
        
        # Calculate IRR for trading period:
def amzn_irr(start_date = None, end_date = None):
    trades = pd.read_csv(f'trades.csv')
    trades['Date'] = pd.to_datetime(trades['Date'])
    if start_date == None:
        start = min(trades['Date'])
    else:
        start = pd.to_datetime(start_date)
    if end_date == None:
        end = max(trades['Date'])
    else:
        end = pd.to_datetime(end_date)
    date_range = pd.date_range(start = start,end=end)
    irr = pd.DataFrame(date_range, columns = ['Date'])
    irr = pd.merge(irr, trades[['Date', 'trade']], how = 'left', on = 'Date')
    irr.trade.fillna(0, inplace =True)
    irr.set_index('Date', drop = True, inplace = True)
    irr_dataframe = irr
    irr = irr['trade'].agg(np.irr)
    irr = irr*365
    return irr 