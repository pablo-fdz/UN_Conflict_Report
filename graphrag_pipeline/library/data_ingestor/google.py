from typing import Tuple, List, Dict, Optional
import polars as pl
import random
import time
import httpx
import trafilatura
import base64
import json
import os
from datetime import datetime
from pathlib import Path 

from pygooglenews import GoogleNews
import pprint
from itertools import islice
from googlenewsdecoder import gnewsdecoder
from datetime import datetime, timedelta
import polars as pl
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed



class GoogleNewsIngestor:
    """
    Class containing the Gogle News data ingestion. It is initialized with:
        region (str): the country or region for which we are writing the report -- acts as the search term in the GN query
        date_interval (tuple): two dates, starting and end. It's the interval for which the GN search is carried out.

    Other class attributes:
        df (pl.DataFrame): dataframe that will hold the news texts and their metadata. Refer to https://docs.google.com/document/d/1zmnxfAxCnyALWReEeEodhm7Hg3VgAsKUsmKBiZdcZxc/edit?tab=t.0 to see the columns it should have.
    """
    def __init__(self, country: str, start_date: str, end_date: str, query_language: str = 'en', query_country: str = 'US'):

        ### INITIALIZE CLASS WITH INFO FROM THE CONFIGURATION FILE ###
        self.search_term = country

        self.query_language = query_language
        self.query_country = query_country
        self.gn = GoogleNews(lang = self.query_language, country = self.query_country)

        self.start_date = start_date
        self.end_date = end_date

        self.USER_AGENTS = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/113.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/114.0.0.0 Safari/537.36",
                "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 Safari/604.1",
                "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15 Safari/604.1",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
                "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0",
                "Mozilla/5.0 (Linux; Android 13; Pixel 7 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36",
                "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1",
                "Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.199 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.5790.110 Safari/537.36",
                "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            ]
        
        self.error_batches = []
        self.expanded_batches = []
        
    # ==================================================================================================
    # ----------------------------------- GOOGLE NEWS QUERY METHODS ------------------------------------
    # ==================================================================================================

    def _date_range_batches(self, start_date, end_date, days_per_batch: int = 3):
        if isinstance(start_date, str):
            start = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start = start_date
        if isinstance(end_date, str):
            end = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end = end_date

        current = start
        while current < end:
            yield current, min(current + timedelta(days=days_per_batch), end)
            current += timedelta(days=days_per_batch)

    def _get_articles(self, start_date, end_date):
        try:
            query = self.gn.search(query=self.search_term, from_=start_date, to_=end_date)
            return query["entries"]
        except Exception as e:
            self.error_batches.append((start_date, end_date))
            return []
        
    def _fetch_batch(self, batch_start, batch_end, days_per_batch):
        start_str = batch_start.strftime("%Y-%m-%d")
        end_str = batch_end.strftime("%Y-%m-%d")
        articles = self._get_articles(start_str, end_str)

        if len(articles) >= 100 and days_per_batch > 1:
            self.expanded_batches.append((start_str, end_str))
            return self._collect_articles(batch_start, batch_end, 1, show_progress=False)
        return articles

    def _build_dataframes(self, articles):
        if not articles:
            print("No articles found for the given query.")
            return
        rows = [{
                    "title": article.get("title"),
                    "google_link": article.get("link", None),
                    "published": article.get("published"),
                    "source": article.get("source", {}).get("title") if article.get("source") else None,
                }
                for article in articles
            ]
        


        df = pl.DataFrame(rows).unique(subset=["google_link"])
        self.n_articles = df.height

        # TO GENERATE THE IDs
        prefix = self.search_term.replace(" ", "")[:3].upper()  # e.g., "SUD"
        random.seed(42)
        unique_digits = random.sample(range(10000, 99999), self.n_articles)

        df = df.with_columns(
                [
                    pl.Series(
                        name="id",
                        values=[f"GN_{prefix}{num}" for num in unique_digits]
                    ),
                    pl.col("published").str.strptime(pl.Date, "%a, %d %b %Y %H:%M:%S %Z", strict=False).alias("date")
                ]
            ).drop("published")
        
        self._split_and_store_dfs(df)
    
    def _split_and_store_dfs(self, df, chunk_size=500):
        n_chunks = (df.height + chunk_size - 1) // chunk_size  # Ceiling division
        for i in range(n_chunks):
            chunk = df.slice(i * chunk_size, chunk_size)
            setattr(self, f"df_{i+1}", chunk)
        self.n_dfs = n_chunks

    def _collect_articles(self, start_date, end_date, days_per_batch, show_progress=True):
        articles = []
        batches = self._date_range_batches(start_date, end_date, days_per_batch)

        with ThreadPoolExecutor(max_workers=64) as executor:
            future_to_range = {
                executor.submit(self._fetch_batch, start, end, days_per_batch): (start, end)
                for start, end in batches
            }
            
            iterator = as_completed(future_to_range)
            if show_progress:
                iterator = tqdm(iterator, total=len(future_to_range), desc="Fetching batches", dynamic_ncols=False, leave=True)

            for future in iterator:
                batch_results = future.result()
                articles.extend(batch_results)

        return articles

    def get_google_news_data(self):
        start_date = self.start_date
        end_date = self.end_date
        days_per_batch = 3  # Default batch size

        articles = self._collect_articles(start_date, end_date, days_per_batch)
        self._build_dataframes(articles)
    
    def print_query_summary(self):
        if isinstance(self.start_date, str):
            start = datetime.strptime(self.start_date, '%Y-%m-%d')
        else:
            start = self.start_date
        if isinstance(self.end_date, str):
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
        else:
            end = self.end_date

        print('------------------------------------------------------')
        print('TOTAL NUMBER OF ARTICLES:' , self.n_articles)  
        print('Average articles per day:', round(self.n_articles / ((end - start).days + 1)))
        print(f'Batches that were expanded to 1-day queries ({len(self.expanded_batches)}):')
        for batch in self.expanded_batches:
            print(f"  --> {batch[0]} to {batch[1]}")
        print(f'Batches that failed ({len(self.error_batches)})')
        print('------------------------------------------------------')
        print(self.df_1.head())

    # ==================================================================================================
    # ------------------------------- URL DECODING + ARTICLE FETCHING ----------------------------------
    # ==================================================================================================

    def _decode_one_url(self, idx_url_tuple):
        idx, url = idx_url_tuple
        interval_time = 1  # interval is optional, default is None
        #proxy = "http://user:pass@localhost:8080" # proxy is optional, default is None
        # proxy = "http://user:pass@proxyhost:port"
        try:
            decoded_url = gnewsdecoder(url, interval=interval_time)
            time.sleep(1.5)
            if decoded_url.get("status"):
                return (idx, decoded_url["decoded_url"])
            else:
                tqdm.write(f"Error: {decoded_url['message']}")
        except Exception as e:
            tqdm.write(f"Error occurred: {e}")
        return (idx, None)
    
    def _decode_urls_concurrently(self, indexed_url_list, max_workers=4):
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._decode_one_url, (idx, url)): idx for idx, url in indexed_url_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc='decoding URLs', dynamic_ncols=False, leave=True):
                result = future.result() # (index, decoded_url)
                if result:
                    results.append(result)
        return results # results will be a list of tuples (index, decoded_url)

    def _fetch_one_article(self, idx_url_tuple, timeout=15):
        idx, url = idx_url_tuple
        headers = {"User-Agent": random.choice(self.USER_AGENTS)}
        proxies = {"http://": "http://user:pass@proxyhost:port",
                   "https://": "http://user:pass@proxyhost:port"}
        try:
            with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout) as client:
                response = client.get(url)
                if response.status_code == 200:
                    html = response.text
                    article_text = trafilatura.extract(html)
                    return (idx, article_text)
        except Exception as e:
            tqdm.write(f"[{idx}] Error fetching article: {e}")
        return (idx, None)

    ### WRAPPER METHOD FOR URL DECODING AND ARTICLE FETCHING ###
    def process_data(self):
        for key, df in self.__dict__.items():
            if isinstance(df, pl.DataFrame) and "google_link" in df.columns:
                indexed_urls = list(enumerate(df["google_link"].to_list()))
                decoded_urls = self._decode_urls_concurrently(indexed_urls)
                
                decoded_dict = dict(decoded_urls)
                final_decoded_list = [decoded_dict.get(i, None) for i in range(len(df))]

                df = df.with_columns(pl.Series(name="decoded_url", values=final_decoded_list))

                # NOW FETCH THE TEXTS
                full_text_tuples = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(self._fetch_one_article, (idx, url)): idx for idx, url in decoded_urls}
                    full_text_tuples = [future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching", ncols=80)]

                full_texts_dict = dict(full_text_tuples)
                final_full_texts = [full_texts_dict.get(i, None) for i in range(len(df))]
                df = df.with_columns(pl.Series(name="full_text", values=final_full_texts))
                df = df.unique(subset=["decoded_url"])
                self.__dict__[key] = df

    def print_urls_and_texts_summary(self):
        for key, df in self.__dict__.items():
            if isinstance(df, pl.DataFrame) and "decoded_url" in df.columns:
                print(f"DataFrame: {key}")
                # Count non-null and non-empty full_texts
                non_nulls = df["full_text"].is_not_null().sum()
                # Robust: handle Null dtype
                if df["full_text"].dtype == pl.Null:
                    non_empty = 0
                else:
                    non_empty = (df["full_text"].is_not_null() & (df["full_text"].cast(str) != '')).sum()
                print(f"Number of correctly fetched articles (non-null): {non_nulls} / {df.height}")
                print(f"Number of non-empty articles: {non_empty} / {df.height}")

    ### SAVE DATA ###
    def save_data(self):
        """
        Exports all DataFrames to Parquet files in the specified directory.
        """
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'google_news'
        output_dir.mkdir(parents=True, exist_ok=True)

        dfs = [df for key, df in self.__dict__.items() if isinstance(df, pl.DataFrame) and "date" in df.columns]
        full_df = pl.concat(dfs, how="vertical").unique(subset=["decoded_url"])
        full_df = full_df.sort("date")

        self._split_and_store_dfs(full_df)

        date_tuple_counts = {}

        for i in range(1, self.n_dfs + 1):
            df = getattr(self, f"df_{i}")

            first_date = str(df["date"].min())
            last_date = str(df["date"].max())
            
            date_tuple = (first_date, last_date)
            suffix = date_tuple_counts.get(date_tuple, 0)

            file_path = output_dir / f"google_news_{self.search_term}_{first_date}_{last_date}_{suffix}.parquet"
            df.write_parquet(file_path)

            date_tuple_counts[date_tuple] = suffix + 1