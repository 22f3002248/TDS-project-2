import json
import os
import re
import urllib
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlencode

import feedparser
import pandas as pd
import pdfplumber
import requests
import tabula
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from geopy.geocoders import Nominatim

# Load environment variables from .env file
load_dotenv()

# Access the variables
token = os.getenv("GITHUB")


def ga_4_1(task):
    pass


def ga_4_2(min_rating=3.0, max_rating=5.0):
    """
    Scrapes IMDb's advanced search results for movies with a rating between min_rating and max_rating,
    and returns up to the first 25 movies in JSON format.

    Each movie is represented as a dictionary with keys:
      - id: The IMDb ID (e.g. tt1234567)
      - title: The movie title
      - year: The release year
      - rating: The movie's rating as a string
    """
    # IMDb advanced search URL with rating filter and count=25 to limit results
    url = f"https://www.imdb.com/search/title/?user_rating={min_rating},{max_rating}&count=25"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.google.com/",
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ConnectionError(
            f"Failed to fetch data from IMDb. Status code: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Locate movie items using the typical IMDb selector
    movie_items = soup.find_all("div", class_="lister-item mode-advanced")
    if not movie_items:
        # Optionally, print a snippet for debugging if needed:
        # print(soup.prettify()[:1000])
        return json.dumps([], indent=2, ensure_ascii=False)

    movies = []
    for item in movie_items[:25]:
        header = item.find("h3", class_="lister-item-header")
        if header is None:
            continue

        title_tag = header.find("a")
        title = title_tag.get_text(strip=True) if title_tag else "Unknown"

        # Extract IMDb id from the href attribute
        href = title_tag["href"] if title_tag and "href" in title_tag.attrs else ""
        movie_id_match = re.search(r"/title/(tt\d+)/", href)
        imdb_id = movie_id_match.group(1) if movie_id_match else "Unknown"

        # Extract year (it may contain extra characters, so we just use the text as-is)
        year_tag = header.find("span", class_="lister-item-year")
        year = year_tag.get_text(strip=True) if year_tag else "Unknown"

        # Extract rating from the rating block; often the numeric rating is stored in data-value attribute
        rating_tag = item.find(
            "div", class_="inline-block ratings-imdb-rating")
        rating = rating_tag["data-value"] if rating_tag and rating_tag.has_attr(
            "data-value") else None
        try:
            rating_float = float(rating) if rating is not None else None
        except ValueError:
            rating_float = None

        if rating_float is not None and min_rating <= rating_float <= max_rating:
            movies.append({
                "id": imdb_id,
                "title": title,
                "year": year,
                "rating": rating
            })

    return json.dumps(movies, indent=2, ensure_ascii=False)


def ga_4_3():
    code = '''
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
import markdown
import uvicorn

app = FastAPI()

# Allow CORS from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_wikipedia_url(country: str) -> str:
    """
    Given a country name, returns the Wikipedia URL for the country.
    """
    return f"https://en.wikipedia.org/wiki/{country}"

def extract_headings_from_html(html: str) -> list:
    """
    Extract all headings (H1 to H6) from the given HTML and return a list.
    """
    soup = BeautifulSoup(html, "html.parser")
    headings = []

    # Loop through all the heading tags (H1 to H6)
    for level in range(1, 7):
        for tag in soup.find_all(f'h{level}'):
            headings.append((level, tag.get_text(strip=True)))

    return headings

def generate_markdown_outline(headings: list) -> str:
    """
    Converts the extracted headings into a markdown-formatted outline.
    """
    markdown_outline = "## Contents\n\n"
    for level, heading in headings:
        markdown_outline += "#" * level + f" {heading}\n\n"
    return markdown_outline

@app.get("/api/outline")
async def get_country_outline(country: str):
    """
    API endpoint that returns the markdown outline of the given country Wikipedia page.
    """
    if not country:
        raise HTTPException(status_code=400, detail="Country parameter is required")

    # Fetch Wikipedia page
    url = get_wikipedia_url(country)
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=404, detail=f"Error fetching Wikipedia page: {e}")

    # Extract headings and generate markdown outline
    headings = extract_headings_from_html(response.text)
    if not headings:
        raise HTTPException(status_code=404, detail="No headings found in the Wikipedia page")

    markdown_outline = generate_markdown_outline(headings)
    return JSONResponse(content={"outline": markdown_outline})

    '''
    return code


def ga_4_4(task):
    """
    Extracts the city from the task description and fetches the weather forecast.

    :param task: Task description containing the city name.
    :return: JSON object with weather forecast.
    """
    # Extract city
    city_match = re.search(
        r'weather forecast description for ([A-Za-z\s]+)', task, re.IGNORECASE)
    if not city_match:
        return "Error: Could not extract city from the task."

    required_city = city_match.group(1).strip()
    # print(f"Fetching weather forecast for: {required_city}")

    # Construct BBC location URL
    location_url = 'https://locator-service.api.bbci.co.uk/locations?' + urlencode({
        'api_key': 'AGbFAKx58hyjQScCXIYrxuEwJh2W2cmv',
        's': required_city,
        'stack': 'aws',
        'locale': 'en',
        'filter': 'international',
        'place-types': 'settlement,airport,district',
        'order': 'importance',
        'a': 'true',
        'format': 'json'
    })

    # Fetch location data
    location_response = requests.get(location_url)
    if location_response.status_code != 200:
        return "Error: Failed to fetch location data."
    result = location_response.json()

    if 'response' not in result or 'results' not in result['response'] or not result['response']['results']['results']:
        return "Error: No location found for the given city."

    location_id = result['response']['results']['results'][0]['id']
    weather_url = f'https://www.bbc.com/weather/{location_id}'

    # Fetch weather data
    weather_response = requests.get(weather_url)
    if weather_response.status_code != 200:
        return "Error: Failed to fetch weather data."

    soup = BeautifulSoup(weather_response.content, 'html.parser')
    daily_summary = soup.find('div', attrs={'class': 'wr-day-summary'})
    if not daily_summary:
        return "Error: Could not extract weather summary."

    daily_summary_list = re.findall('[a-zA-Z][^A-Z]*', daily_summary.text)

    # Generate date list
    datelist = pd.date_range(
        datetime.today(), periods=len(daily_summary_list)).tolist()
    datelist = [date.date().strftime('%Y-%m-%d') for date in datelist]

    # Map dates to descriptions
    weather_data = {date: desc for date,
                    desc in zip(datelist, daily_summary_list)}

    # Convert to JSON
    return weather_data


# Example usage
# task_description = "What is the JSON weather forecast description for Kinshasa?"
# print(get_weather_forecast(task_description))


def ga_4_5(task):
    """
    Extracts city, country, and required coordinate type (latitude/longitude) from the task, 
    then retrieves the requested coordinate using Nominatim.

    :param task: Task description containing the city, country, and coordinate type.
    :return: The requested coordinate value (minimum or maximum) from the bounding box.
    """

    # Extract bound type (minimum or maximum)
    bound_match = re.search(r"\b(minimum|maximum)\b", task, re.IGNORECASE)
    bound_type = bound_match.group(1).lower() if bound_match else None

    # Extract coordinate type (latitude or longitude)
    coord_match = re.search(r"\b(latitude|longitude)\b", task, re.IGNORECASE)
    coord_type = coord_match.group(1).lower() if coord_match else None

    # Extract city
    city_match = re.search(
        r"\bthe city ([A-Za-z\s]+?) in the", task, re.IGNORECASE)
    city = city_match.group(1).strip() if city_match else None

    # Extract country
    country_match = re.search(
        r"\bthe country ([A-Za-z\s]+?) on the Nominatim API", task, re.IGNORECASE)
    country = country_match.group(1).strip() if country_match else None

    # Ensure all parameters are extracted
    if not all([bound_type, coord_type, city, country]):
        return "Error: Could not extract required details from the task."

    print(
        f"City: {city}, Country: {country}, Coordinate Type: {coord_type}, Bound Type: {bound_type}")

    # Activate the Nominatim geocoder
    locator = Nominatim(user_agent="myGeocoder")

    # Geocode the city and country
    location = locator.geocode(f"{city}, {country}")

    # Check if the location was found
    if location:
        bounding_box = location.raw.get('boundingbox', [])

        # Format: [min_lat, max_lat, min_lon, max_lon]
        if len(bounding_box) == 4:
            if coord_type == "latitude":
                coordinate = bounding_box[0] if bound_type == "minimum" else bounding_box[1]
            else:  # coord_type == "longitude"
                coordinate = bounding_box[2] if bound_type == "minimum" else bounding_box[3]

            return str(coordinate)
        else:
            return "Bounding box information not available."
    else:
        return "Location not found."


# Example usage
# task_description = "What is the maximum latitude of the bounding box of the city Cairo in the country Egypt on the Nominatim API?"
# print(ga_4_5(task_description))


def ga_4_6(task):
    """
    Searches Hacker News RSS for the latest post mentioning a specific topic with a minimum number of points.

    :param task: Task description containing topic and points.
    :return: The URL of the latest valid Hacker News post in JSON format.
    """
    # Extract the topic and minimum points from the task
    topic_match = re.search(r"mentioning (.*?) and", task)
    points_match = re.search(r"a minimum of (\d+) points", task)
    if not topic_match or not points_match:
        return json.dumps({"answer": "Invalid task format"})

    topic = topic_match.group(1).strip()
    min_points = int(points_match.group(1))

    # Properly encode the topic for the URL
    encoded_topic = urllib.parse.quote(topic)

    # Fetch latest Hacker News posts using HNRSS API
    feed_url = f"https://hnrss.org/newest?q={encoded_topic}&points={min_points}"
    feed = feedparser.parse(feed_url)

    # Extract the link of the latest post
    latest_post_link = feed.entries[0].link if feed.entries else "No posts found."

    return latest_post_link


# Example Usage
# task_description = """
# Search using the Hacker News RSS API for the latest Hacker News post mentioning Hacker Culture and having a minimum of 32 points.
# What is the link that it points to?
# """

# # Call function
# result = ga_4_6(task_description)
# print(result)


def ga_4_7(task):
    """
    Extracts city and minimum followers from the task description and queries the GitHub API
    to find the newest user in that city with at least the specified number of followers.

    :param task: The task description containing city and follower count.
    :param token: GitHub personal access token.
    :return: The ISO 8601 creation date of the newest valid user.
    """
    # Extract city and minimum followers using regex
    city_match = re.search(r'located in the city (\w+)', task)
    followers_match = re.search(r'over (\d+) followers', task)
    ignore_date_match = re.search(
        r'after (\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2}:\d{2} [APM]+)', task)

    if not city_match or not followers_match or not ignore_date_match:
        raise ValueError(
            "Could not extract city, follower count, or ignore date from the task.")

    city = city_match.group(1)  # Extract city name
    # Extract minimum follower count
    min_followers = int(followers_match.group(1))

    # Convert ignore date to datetime object
    ignore_date_str = ignore_date_match.group(1)
    ignore_date_obj = datetime.strptime(
        ignore_date_str, "%m/%d/%Y, %I:%M:%S %p")

    # GitHub API search query (sorted by newest joined date)
    url = f"https://api.github.com/search/users?q=location:{city}+followers:>{min_followers}&sort=joined&order=desc"

    # Send API request
    headers = {"Authorization": f"token {token}",
               "Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"GitHub API Error: {response.status_code}, {response.text}")

    users = response.json().get("items", [])

    # Find the newest valid user (who joined before the ignore_date)
    for user in users:
        user_url = user["url"]  # Get user profile URL

        # Fetch full user details
        user_response = requests.get(user_url, headers=headers)
        if user_response.status_code != 200:
            continue

        user_data = user_response.json()
        created_at = user_data.get("created_at")

        # Convert created_at to datetime object
        user_created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")

        # Check if the user joined before the ignore_date
        if user_created_date < ignore_date_obj:
            return created_at  # Return the first valid user's creation date

    return "No eligible users found before the ignore date."


# Example Usage:
# task_description = """
# Your Task
# Using the GitHub API, find all users located in the city Beijing with over 60 followers.

# When was the newest user's GitHub profile created?

# API Integration and Data Retrieval: Leverage GitHub’s search endpoints to query users by location and filter them by follower count.
# Data Processing: From the returned list of GitHub users, isolate those profiles that meet the specified criteria.
# Sort and Format: Identify the "newest" user by comparing the created_at dates provided in the user profile data. Format the account creation date in the ISO 8601 standard (e.g., "2024-01-01T00:00:00Z").

# Enter the date (ISO 8601, e.g. "2024-01-01T00:00:00Z") when the newest user joined GitHub.
# Search using location: and followers: filters, sort by joined descending, fetch the first url, and enter the created_at field. Ignore ultra-new users who JUST joined, i.e. after 3/29/2025, 10:21:43 AM.
# """

# Call function (Replace 'your-github-token' with a valid token)
# response = ga_4_7(task_description)
# print(response)


def ga_4_8(task):
    """
    Extracts the email from the task description and sets up a scheduled GitHub Action
    workflow using the GitHub API. The workflow runs daily and commits a file.

    :param task: The task description containing the email.
    :param token: GitHub personal access token (with repo and workflow permissions).
    :return: Response from the GitHub API.
    """
    # Extract repository URL and email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', task)

    if not email_match:
        email = "22f3002248@ds.study.iitm.ac.in"
    else:
        email = email_match.group()

    # GitHub API URL for creating a workflow file
    url = f"https://api.github.com/repos/22f3002248/tds-project-2-action-2/contents/.github/workflows/daily_commit.yml"

    # Define the GitHub Actions workflow (YAML content)
    workflow_content = f"""name: Daily Commit

on:
  schedule:
    - cron: '0 0 * * *'  # Runs at midnight UTC every day

jobs:
  commit-job:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Update file - {email}
        run: |
          echo "$(date)" >> daily_log.txt

      - name: Commit and Push Changes
        run: |
          git config --local user.name "GitHub Action"
          git config --local user.email "{email}"
          git add daily_log.txt
          git commit -m "Daily commit $(date)"
          git push
    """

    # Encode content in base64 for GitHub API
    import base64
    encoded_content = base64.b64encode(workflow_content.encode()).decode()

    # Prepare request payload
    payload = {
        "message": "Add daily commit workflow",
        "content": encoded_content,
        "branch": "main"  # Change if using a different branch
    }

    # Send API request
    headers = {"Authorization": f"token {token}",
               "Accept": "application/vnd.github.v3+json"}
    response = requests.put(url, json=payload, headers=headers)
    print(response)
    return "https://api.github.com/repos/22f3002248/tds-project-2-action-2"


# Example usage:
# task_description = """
# Your Task
# Create a scheduled GitHub action that runs daily and adds a commit to your repository.
# The workflow should:

# - Use schedule with cron syntax to run once per day (must use specific hours/minutes, not wildcards)
# - Include a step with your email 22f3002248@ds.study.iitm.ac.in in its name
# - Create a commit in each run
# - Be located in .github/workflows/ directory

# After creating the workflow:
# - Trigger the workflow and wait for it to complete
# - Ensure it appears as the most recent action in your repository
# - Verify that it creates a commit during or within 5 minutes of the workflow run

# Enter your repository URL (format: https://github.com/USER/REPO):
# """

# # Call function (Replace 'your-github-token' with a valid token)
# response = ga_4_8(task_description)
# print(response)


def ga_4_9(pdf_path, task):
    """
    Extracts student marks data from a PDF, filters based on the given task criteria, 
    and calculates the total marks for the specified subject.

    :param pdf_path: Path to the PDF file.
    :param task: The input task description containing filtering criteria.
    :return: The computed total marks based on the extracted criteria.
    """

    # Extract key parameters dynamically from the task description
    match = re.search(
        r'total (\w+) marks of students who scored (\d+) or more marks in (\w+) in groups (\d+)-(\d+)',
        task, re.IGNORECASE
    )

    if not match:
        raise ValueError(
            "Could not extract parameters from the task description.")

    target_subject = match.group(1).capitalize()  # e.g., "Physics"
    min_marks = int(match.group(2))  # e.g., 29
    filter_subject = match.group(3).capitalize()  # e.g., "Maths"
    group_min = int(match.group(4))  # e.g., 23
    group_max = int(match.group(5))  # e.g., 55

    # Extract tables from the PDF
    tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
    all_dfs = []

    for i, table in enumerate(tables):
        table["Group"] = i + 1
        all_dfs.append(table)

    df = pd.concat(all_dfs, ignore_index=True)

    # Standardize column names
    df.columns = ["Maths", "Physics", "English",
                  "Economics", "Biology", "Group"]

    # Convert columns to numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    # Apply filtering conditions dynamically based on extracted parameters
    filtered_df = df[(df[filter_subject] >= min_marks) &
                     df["Group"].between(group_min, group_max)]

    # Compute the total marks for the specified subject
    total_marks = filtered_df[target_subject].sum()

    return total_marks


# Example usage:
# task = """
# This file, <q-extract-tables-from-pdf.pdf> contains a table of student marks in Maths, Physics, English, Economics, and Biology.

# Calculate the total Physics marks of students who scored 29 or more marks in Maths in groups 23-55 (including both groups).
# """

# result = ga_4_9("data/q-extract-tables-from-pdf.pdf", task)
# print(result)


def ga_4_10(pdf_path):
    """
    Extracts text from a PDF and converts it into Markdown format, preserving headings, lists, and tables.

    :param pdf_path: Path to the PDF file.
    :return: Formatted Markdown text.
    """
    markdown_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split("\n")

                for line in lines:
                    line = line.strip()

                    # Convert headings based on capitalization pattern
                    if re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+)+$", line):
                        markdown_text.append(f"# {line}")
                    elif re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+)+:", line):
                        markdown_text.append(f"## {line[:-1]}")
                    elif re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+)+\.", line):
                        markdown_text.append(f"### {line[:-1]}")

                    # Convert bullet points (replace special characters with '-')
                    elif re.match(r"^[•-] ", line):
                        markdown_text.append(f"- {line[2:]}")

                    # Detect tables (lines with '|' and '-')
                    elif "|" in line and "---" not in line:
                        markdown_text.append(line)

                    # Preserve regular text
                    else:
                        markdown_text.append(line)

    return "\n\n".join(markdown_text)


# Example usage
# pdf_path = "data/q-pdf-to-markdown.pdf"  # Replace with the actual file path
# extracted_text = ga_4_10(pdf_path)
# print(extracted_text)
