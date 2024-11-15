import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import swisseph as swe
from datetime import datetime
from geopy.geocoders import Nominatim
import pytz
import streamlit as st
import re
from timezonefinder import TimezoneFinder
from geopy import geocoders
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import json
import openai
import hashlib
import time
from geopy.exc import GeocoderTimedOut
import geopy
from sklearn.preprocessing import MinMaxScaler
from unidecode import unidecode
import base64
import streamlit.components.v1 as components

with open("google_analytics.html", "r") as f:
    html_code = f.read()
    components.html(html_code, height=0)

# Đường dẫn tương đối tới thư mục ephemeris (trong cùng thư mục với file Python chính)
relative_ephe_path = os.path.join(os.path.dirname(__file__), 'sweph')
# Đặt đường dẫn tới thư viện Swiss Ephemeris
swe.set_ephe_path(relative_ephe_path)

# Đọc dữ liệu từ các file CSV
financial_traits_path = 'combined_financial_traits.csv'
keyword_to_trait_mapping_df = pd.read_csv('keyword_to_trait_mapping.csv')
product_keywords_path = 'product_keywords.csv'
aspect_path = 'aspect_sc.csv'

# Read data
financial_traits_df = pd.read_csv(financial_traits_path)
keyword_to_trait_mapping = keyword_to_trait_mapping_df.set_index('Keyword').T.to_dict()
product_keywords_df = pd.read_csv(product_keywords_path)
product_keywords = product_keywords_df.set_index('Product').T.to_dict('list')
aspect_influence_factors_df = pd.read_csv(aspect_path)

# Rulers for each zodiac sign
rulers = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": ["Mars", "Pluto"],
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": ["Saturn", "Uranus"],
    "Pisces": ["Jupiter", "Neptune"]
}
# Ảnh hưởng của hành tinh đối với hành vi tài chính dựa trên khoảng cách từ các hành tinh
planet_impacts = {
    'Venus': 0.20,
    'Jupiter': 0.15,
    'Saturn': 0.15,
    'Mars': 0.10,
    'Pluto': 0.10,
    'Mercury': 0.05
}

# Vị trí của các hành tinh trong cung Hoàng Đạo
zodiac_positions = {
    'Sun': 10,  
    'Moon': 7,
    'Mercury': 5,
    'Mars': 8,
    'Venus': 6,
    'Jupiter': 9,
    'Saturn': 7,
    'Uranus': 4,
    'Neptune': 3,
    'Pluto': 2
}

# Function to calculate zodiac sign and degree
def get_zodiac_sign_and_degree(degree):
    zodiacs = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ]
    sign_index = int(degree / 30)
    sign = zodiacs[sign_index]
    degree_in_sign = degree % 30
    return sign, degree_in_sign

# Đường dẫn tới file cache
CITY_CACHE_FILE = 'city_cache.txt'
LOCATION_CACHE_FILE = 'location_cache.txt'

# Hàm lưu cache của city_suggestions vào file
def save_city_cache_to_file(place, city_suggestions):
    with open(CITY_CACHE_FILE, "a") as f:
        f.write(f"{place}|{city_suggestions}\n")

# Hàm đọc cache city_suggestions từ file
def read_city_cache():
    cache = {}
    try:
        with open("city_cache.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                # Kiểm tra xem dòng có thể phân tách thành 2 phần tử hay không
                parts = line.strip().split("|")
                if len(parts) == 2:
                    place, city_suggestions = parts
                    cache[place] = eval(city_suggestions)  # Convert the string list back to a list
                else:
                    # Nếu dòng không đủ dữ liệu, bỏ qua hoặc ghi log cảnh báo
                    print(f"Invalid cache entry: {line}")
    except FileNotFoundError:
        pass
    return cache


# Hàm lưu cache của lat, lon, timezone vào file
def save_location_cache_to_file(place, lat, lon, timezone):
    with open(LOCATION_CACHE_FILE, "a") as f:
        f.write(f"{place}|{lat}|{lon}|{timezone}\n")

# Hàm đọc cache lat, lon, timezone từ file
def read_location_cache():
    cache = {}
    try:
        with open("location_cache.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split("|")
                # Kiểm tra nếu dòng có đúng 4 phần tử thì mới tiến hành xử lý
                if len(parts) == 4:
                    place, lat, lon, timezone = parts
                    cache[place] = {
                        'lat': float(lat),
                        'lon': float(lon),
                        'timezone': timezone
                    }
                else:
                    # Nếu dòng không đủ dữ liệu, ghi log cảnh báo và bỏ qua dòng đó
                    print(f"Invalid cache entry: {line}")
    except FileNotFoundError:
        pass
    return cache

def normalize_place(place):
    # Loại bỏ dấu và chuyển tất cả về chữ thường, đồng thời loại bỏ khoảng trắng
    normalized = unidecode(place).replace(" ", "").lower()
    # Loại bỏ các ký tự không phải là chữ cái (A-Z)
    normalized = re.sub(r'[^a-z]', '', normalized)
    return normalized

# Đọc cả hai loại cache từ file
city_cache = read_city_cache()
location_cache = read_location_cache()

# Hàm lấy city_suggestions từ cache hoặc API
def get_city_suggestions(query):
    if not query or not query.strip():
        # Kiểm tra ngôn ngữ người dùng
        if language == "Tiếng Việt":
            return ["Vui lòng nhập vào nơi sinh của bạn"]
        else:
            return ["Please input your birthplace"]
    
    normalized_place = normalize_place(query)

    # Kiểm tra xem place đã có trong cache chưa
    if normalized_place in city_cache:
        return city_cache[normalized_place]

    # Nếu không có trong cache hoặc cache trống, gọi API
    geolocator = Nominatim(user_agent="astrology_app")
    time.sleep(1)  # Chờ một giây để tránh giới hạn API
    try:
        location = geolocator.geocode(query, exactly_one=False, limit=5, language='en')
        if location:
            result = [f"{loc.address} ({loc.latitude}, {loc.longitude})" for loc in location]
            city_cache[normalized_place] = result
            save_city_cache_to_file(normalized_place, result)
            return result
        else:
            st.warning("No matching city found.")
    except Exception as e:
        if normalized_place in city_cache:  # Nếu cache đã có place, hiển thị từ cache
            return city_cache[normalized_place]
        st.error(f"Error occurred: {e}")
    return []


# Hàm lấy lat, lon, timezone từ cache hoặc API
def get_location_and_timezone(place):
    normalized_place = normalize_place(place)

    # Kiểm tra cache trước
    if normalized_place in location_cache:
        cached_data = location_cache[normalized_place]
        lat = cached_data.get('lat')
        lon = cached_data.get('lon')
        timezone = cached_data.get('timezone')

        # Nếu cache có đầy đủ dữ liệu, trả về kết quả từ cache
        if lat is not None and lon is not None and timezone is not None:
            return lat, lon, timezone
        else:
            st.warning(f"Cache data for {place} is invalid. Retrieving fresh data.")
    
    # Nếu cache không có hoặc dữ liệu không hợp lệ, gọi API để lấy dữ liệu mới
    geolocator = Nominatim(user_agent="astrology_app")
    location = geolocator.geocode(place)  # Gọi API với chuỗi gốc
    
    if location:
        lat, lon = location.latitude, location.longitude
        tf = TimezoneFinder()
        timezone = tf.timezone_at(lat=lat, lng=lon)

        if timezone is None:
            st.error(f"Unable to retrieve timezone for location: {place}. Please try again.")
            return None, None, None

        # Lưu lại kết quả vào cache nếu có dữ liệu hợp lệ
        location_cache[normalized_place] = {
            'lat': lat, 
            'lon': lon, 
            'timezone': timezone
        }
        save_location_cache_to_file(normalized_place, lat, lon, timezone)

        return lat, lon, timezone
    else:
        st.error(f"Cannot find location for place: {place}")
        return None, None, None


# Function to convert decimal degrees to DMS (degrees, minutes, seconds)
def decimal_to_dms(degree):
    d = int(degree)
    m = int((degree - d) * 60)
    s = (degree - d - m / 60) * 3600
    return f"{d}° {m}' {s:.2f}\""

# Function to calculate planetary positions
def get_planet_positions(year, month, day, hour, minute, lat, lon, timezone):
    tz = pytz.timezone(timezone)
    local_datetime = datetime(year, month, day, hour, minute)
    local_datetime = tz.localize(local_datetime)
    utc_datetime = local_datetime.astimezone(pytz.utc)
    jd = swe.julday(utc_datetime.year, utc_datetime.month, utc_datetime.day, utc_datetime.hour + utc_datetime.minute / 60.0)
    
    swe.set_topo(lon, lat, 0)
    planets = {
        'Sun': swe.SUN, 'Moon': swe.MOON, 'Mercury': swe.MERCURY, 'Venus': swe.VENUS,
        'Mars': swe.MARS, 'Jupiter': swe.JUPITER, 'Saturn': swe.SATURN,
        'Uranus': swe.URANUS, 'Neptune': swe.NEPTUNE, 'Pluto': swe.PLUTO,
        'North Node': swe.TRUE_NODE, 'Chiron': swe.CHIRON, 'Lilith': swe.MEAN_APOG,
        'True Lilith': swe.OSCU_APOG, 'Vertex': swe.VERTEX, 'Ceres': swe.CERES,
        'Pallas': swe.PALLAS, 'Juno': swe.JUNO, 'Vesta': swe.VESTA
    }
    
    positions = {}
    for planet, id in planets.items():
        position, _ = swe.calc_ut(jd, id)
        positions[planet] = position[0]
    
    positions['South Node'] = (positions['North Node'] + 180.0) % 360.0
    return positions

# Function to calculate houses and ascendant
def calculate_houses_and_ascendant(year, month, day, hour, minute, lat, lon, timezone):
    tz = pytz.timezone(timezone)
    local_datetime = datetime(year, month, day, hour, minute)
    local_datetime = tz.localize(local_datetime)
    utc_datetime = local_datetime.astimezone(pytz.utc)
    jd = swe.julday(utc_datetime.year, utc_datetime.month, utc_datetime.day,
                    utc_datetime.hour + utc_datetime.minute / 60.0 + utc_datetime.second / 3600.0)
    
    # House calculation using Placidus system
    houses, ascmc = swe.houses(jd, lat, lon, b'P')
    ascendant = ascmc[0]
    house_cusps = {f'House {i+1}': houses[i] for i in range(12)}
    return ascendant, house_cusps

# Function to determine which house a planet is in
def get_house_for_planet(planet_degree, houses):
    house_degrees = [houses[f'House {i+1}'] for i in range(12)]
    house_degrees.append(house_degrees[0] + 360)  # wrap-around for the 12th house
    
    for i in range(12):
        start = house_degrees[i]
        end = house_degrees[(i+1) % 12]
        if start < end:
            if start <= planet_degree < end:
                return i + 1  # Trả về số nhà thay vì chuỗi "House X"
        else:  # handle the wrap-around
            if start <= planet_degree or planet_degree < end:
                return i + 1  # Trả về số nhà thay vì chuỗi "House X"
    return 'Unknown'  # Default case if no house matches


# Function to display planetary data with house placement
def create_astrology_dataframe(positions, houses, ascendant):
    planets_info = []

    # Add Ascendant to the list
    asc_sign, asc_degree_in_sign = get_zodiac_sign_and_degree(ascendant)
    asc_dms = decimal_to_dms(asc_degree_in_sign)
    
    # Append Ascendant first (no need for house ruler)
    planets_info.append(["Ascendant", asc_sign, "1", asc_dms])  # Chỉ 4 giá trị

    # Append other planets
    for planet, degree in positions.items():
        sign, degree_in_sign = get_zodiac_sign_and_degree(degree)
        dms = decimal_to_dms(degree_in_sign)
        house_cup = get_house_for_planet(degree, houses)  # Chỉ lấy house cup thay vì house ruler
        planets_info.append([planet, sign, house_cup, dms])  # Chỉ 4 giá trị

    # Đảm bảo là có đúng 4 cột
    df = pd.DataFrame(planets_info, columns=['Planet', 'Zodiac Sign', 'House', 'Degree'])
    return df

def create_house_dataframe(houses, ascendant):
    house_info = []

    # Xác định cung hoàng đạo và độ của Ascendant
    asc_sign, asc_degree_in_sign = get_zodiac_sign_and_degree(ascendant)
    asc_dms = decimal_to_dms(asc_degree_in_sign)
    asc_house_ruler = rulers.get(asc_sign, "Unknown")
    if isinstance(asc_house_ruler, list):
        asc_house_ruler = ', '.join(asc_house_ruler)
    
    # Append Ascendant with Zodiac Sign, Ruler, and Degree
    house_info.append(['Ascendant', asc_sign, asc_house_ruler, asc_dms])  # Sử dụng Ascendant cho nhà 1

    # Append other houses with Zodiac Sign and Ruler
    for i in range(1, 12):
        house_degree = houses[f'House {i+1}']
        house_sign, degree_in_sign = get_zodiac_sign_and_degree(house_degree)
        dms = decimal_to_dms(degree_in_sign)
        house_ruler = rulers.get(house_sign, "Unknown")
        if isinstance(house_ruler, list):
            house_ruler = ', '.join(house_ruler)
        house_info.append([f'House {i+1}', house_sign, house_ruler, dms])

    # Tạo DataFrame từ house_info, bao gồm cột House, House Cup, Ruler và Degree
    df = pd.DataFrame(house_info, columns=['House', 'House Cup', 'Ruler', 'House Cup Degree'])
    return df

# Hàm để tính các góc hợp giữa các hành tinh và góc độ
def calculate_aspects(positions):
    aspects = []
    planets = list(positions.keys())
    aspect_angles = [0, 60, 90, 120, 150, 180]
    aspect_names = ['Conjunction', 'Sextile', 'Square', 'Trine', 'Quincunx', 'Opposition']
    aspect_orbs = [10.5, 6.1, 7.8, 8.3, 2.7, 10]  # Orb for each aspect

    for i, planet1 in enumerate(planets):
        for planet2 in planets[i + 1:]:
            pos1 = positions[planet1]
            pos2 = positions[planet2]
            angle = abs(pos1 - pos2)
            if angle > 180:
                angle = 360 - angle
            for aspect_angle, orb, name in zip(aspect_angles, aspect_orbs, aspect_names):
                if abs(angle - aspect_angle) <= orb:
                    aspects.append((planet1, planet2, name, round(angle, 2)))  # Lưu cả góc độ (angle) đã được làm tròn
                    break
    return aspects

# Hàm để tạo DataFrame cho các góc hợp của người dùng, bổ sung góc độ
def create_aspects_dataframe(aspects, positions):
    aspect_data = []
    for planet1, planet2, aspect_type, degree in aspects:
        # Lấy cung hoàng đạo và độ của hành tinh 1
        planet1_sign, planet1_degree_in_sign = get_zodiac_sign_and_degree(positions[planet1])
        # Lấy cung hoàng đạo và độ của hành tinh 2
        planet2_sign, planet2_degree_in_sign = get_zodiac_sign_and_degree(positions[planet2])
        
        # Thêm dữ liệu vào bảng
        aspect_data.append([planet1, planet1_sign, planet1_degree_in_sign, planet2, planet2_sign, planet2_degree_in_sign, aspect_type, degree])
    
    # Tạo DataFrame với các cột cần thiết
    df = pd.DataFrame(aspect_data, columns=['Planet 1', 'Zodiac Sign 1', 'Degree in Sign 1', 'Planet 2', 'Zodiac Sign 2', 'Degree in Sign 2', 'Aspect', 'Degree'])
    
    return df


# --------------------------------TRAITs-----------------------------------------------
# Hàm xác định hành tinh dẫn dắt
def get_dominant_planet(planet1, planet2, individual_planets):
    if planet1 not in zodiac_positions or planet2 not in zodiac_positions:
        return None  # Bỏ qua nếu hành tinh không trong danh sách

    planet1_power = zodiac_positions.get(planet1, 0)
    planet2_power = zodiac_positions.get(planet2, 0)

    try:
        planet1_sign = [sign for planet, sign in individual_planets if planet == planet1][0]
    except IndexError:
        planet1_sign = None
    
    try:
        planet2_sign = [sign for planet, sign in individual_planets if planet == planet2][0]
    except IndexError:
        planet2_sign = None

    if planet1_sign and (planet1 in rulers.get(planet1_sign, []) if isinstance(rulers.get(planet1_sign), list) else planet1 == rulers.get(planet1_sign)):
        return planet1

    if planet2_sign and (planet2 in rulers.get(planet2_sign, []) if isinstance(rulers.get(planet2_sign), list) else planet2 == rulers.get(planet2_sign)):
        return planet2

    if planet1_power > planet2_power:
        dominant_planet = planet1
    elif planet2_power > planet1_power:
        dominant_planet = planet2
    else:
        dominant_planet = planet1
    
    return dominant_planet
# Định dạng các góc hợp từ bảng aspects
def format_aspects(row, individual_planets):
    # Lấy nội dung từ cột 'Aspects'
    aspects = row['Aspects']
    
    # Sử dụng regular expression để trích xuất planet và aspect type
    aspects_list = df_aspects[['Planet 1', 'Planet 2', 'Aspect']].values.tolist()
    formatted_aspects = []
    for planet1, planet2, aspect_type in aspects_list:
        if planet1 not in zodiac_positions or planet2 not in zodiac_positions:
            continue
        dominant_planet = get_dominant_planet(planet1, planet2, individual_planets)
        if dominant_planet == planet2:
            planet1, planet2 = planet2, planet1
        formatted_aspects.append(f"{planet1} {aspect_type} {planet2}")
    return "\n".join(formatted_aspects)

# Trích xuất các góc hợp liên quan đến các hành tinh đã chọn
def extract_relevant_aspects(formatted_aspects, relevant_planets):
    aspects = re.findall(r"(\w+)\s+(Conjunction|Sextile|Square|Trine|Opposition|Quincunx)\s+(\w+)", formatted_aspects)
    
    # Chỉ giữ lại các góc hợp liên quan đến các hành tinh trong relevant_planets
    filtered_aspects = []
    for aspect in aspects:
        planet1, aspect_type, planet2 = aspect
        if planet1 in relevant_planets and planet2 in relevant_planets:
            filtered_aspects.append((planet1, aspect_type, planet2))
        else:
            print(f"Aspect không hợp lệ: {planet1} hoặc {planet2}")
    
    return filtered_aspects

# Tính toán các đặc điểm tài chính dựa trên vị trí hành tinh và góc hợp
def calculate_financial_traits(individual_planets, formatted_aspects):
    # Danh sách các hành tinh mà bạn muốn lấy
    selected_planets = ['Sun', 'Moon', 'Mercury', 'Mars', 'Venus', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
    individual_planets = [(row['Planet'], row['Zodiac Sign']) for _, row in df_positions.iterrows() if row['Planet'] in selected_planets]

    final_scores = {trait: 0 for trait in ['Adventurous', 'Convenient', 'Impulsive', 'Conservative', 'Cautious', 'Analytical']}

    # Tính điểm dựa trên vị trí của các hành tinh và cung hoàng đạo
    for planet, sign in individual_planets:
        planet_scores_df = financial_traits_df[financial_traits_df['Planet'] == f"{planet} {sign}"]
        if not planet_scores_df.empty:
            planet_scores = planet_scores_df.iloc[0]
            # print(planet_scores)

            for trait in final_scores.keys():
                base_score = planet_scores[trait]
                
                if planet in planet_impacts:
                    base_score += planet_impacts[planet]
                    # print(base_score)

                final_scores[trait] += base_score
    #             print(final_scores)
    # print(final_scores)

    # Khởi tạo tổng ảnh hưởng của các góc hợp cho từng đặc điểm
    aspects = extract_relevant_aspects(formatted_aspects, [planet[0] for planet in individual_planets])
    # print(f"Extracted Aspects: {aspects}")
    
     # Initialize scores
    final_scores = {trait: 0 for trait in ['Adventurous', 'Convenient', 'Impulsive', 'Conservative', 'Cautious', 'Analytical']}

    # Compute base scores from planets and zodiac signs
    for planet, sign in individual_planets:
        planet_scores_df = financial_traits_df[financial_traits_df['Planet'] == f"{planet} {sign}"]
        if not planet_scores_df.empty:
            planet_scores = planet_scores_df.iloc[0]
            for trait in final_scores.keys():
                base_score = planet_scores[trait]
                if planet in planet_impacts:
                    base_score += planet_impacts[planet]
                    # print(base_score)
                final_scores[trait] += base_score
    #             print(final_scores)
    # print(final_scores)

    # Initialize total aspect influence
    aspects = extract_relevant_aspects(formatted_aspects, [planet[0] for planet in individual_planets])
    total_aspect_influence = {trait: 0 for trait in final_scores.keys()}

    # Processing aspects from the CSV and checking if the data is being matched
    for _, row in df_aspects.iterrows():
        planet1, planet2, aspect_type = row['Planet 1'], row['Planet 2'], row['Aspect']
        dominant_planet = get_dominant_planet(planet1, planet2, individual_planets)
        
        if dominant_planet == planet2:
            planet1, planet2 = planet2, planet1

        # Check for matching aspect
        matching_aspect_df = aspect_influence_factors_df[
            (aspect_influence_factors_df['Planet1'] == planet1) &
            (aspect_influence_factors_df['Planet2'] == planet2) &
            (aspect_influence_factors_df['Aspect'] == aspect_type)
        ]
        
        if not matching_aspect_df.empty:
            for trait in final_scores.keys():
                aspect_influence_value = matching_aspect_df.iloc[0].get(trait)
                if aspect_influence_value:
                    aspect_influence = float(aspect_influence_value.strip('%')) / 100
                    total_aspect_influence[trait] += aspect_influence
                    # print(total_aspect_influence)


    # Apply aspect influence to the final scores
    for trait in final_scores.keys():
        adjusted_score = final_scores[trait] + (total_aspect_influence[trait] * 10)
        final_scores[trait] = adjusted_score

    # Normalize final scores to be between 0 and 5
    for trait in final_scores.keys():
        final_scores[trait] /= len(individual_planets)
        final_scores[trait] = min(max(final_scores[trait], 0), 5)
    print(final_scores)
    return final_scores

# -------------------------------------DRAW RADAR CHART----------------------------------
# Dictionary mapping traits from English to Vietnamese
# Tạo từ điển để dịch các traits
trait_translation = {
    "Adventurous": "Mạo Hiểm",
    "Convenient": "Tiện Lợi",
    "Impulsive": "Phóng Khoáng",
    "Conservative": "Kiên Định",
    "Analytical": "Tỉ Mỉ",
    "Cautious": "Cẩn Trọng"
}

# Lấy danh sách traits theo ngôn ngữ đã chọn
def get_traits_by_language(language):
    traits_english = list(trait_translation.keys())  # Danh sách tiếng Anh
    if language == "Tiếng Việt":
        return [trait_translation[trait] for trait in traits_english]  # Chuyển sang tiếng Việt
    else:
        return traits_english  # Trả về tiếng Anh


# Định nghĩa hàm để xác định mức độ dựa trên điểm
def get_score_level(score, language="Tiếng Việt"):
    if language == "Tiếng Việt":
        if 1.00 <= score <= 1.80:
            return "Rất thấp"
        elif 1.81 <= score <= 2.60:
            return "Thấp"
        elif 2.61 <= score <= 3.40:
            return "Trung bình"
        elif 3.41 <= score <= 4.20:
            return "Cao"
        elif 4.21 <= score <= 5.00:
            return "Rất cao"
    elif language == "English":
        if 1.00 <= score <= 1.80:
            return "Incredibly low"
        elif 1.81 <= score <= 2.60:
            return "Low"
        elif 2.61 <= score <= 3.40:
            return "Average"
        elif 3.41 <= score <= 4.20:
            return "High"
        elif 4.21 <= score <= 5.00:
            return "Incredibly high"


# Function to get hover text color based on score level
def get_score_color(score, language="Tiếng Việt"):
    level = get_score_level(score, language)
    if level in ["Rất thấp", "Incredibly low"]:
        return "#ff0000"  # Pastel Red
    elif level in ["Thấp", "Low"]:
        return "#ff5f00"  # Pastel Orange
    elif level in ["Trung bình", "Average"]:
        return "#ffaf00"  # Pastel Yellow
    elif level in ["Cao", "High"]:
        return "#008700"  # Pastel Green
    else:  # "Rất cao" or "Incredibly high"
        return "#005fff"  # Pastel Blue
        
# Hàm vẽ radar chart tương tác với plotly
def plot_radar_chart(final_scores, average_scores, language):
    # Lấy danh sách traits bằng tiếng Anh và theo ngôn ngữ đã chọn
    traits_english = list(trait_translation.keys())
    traits = get_traits_by_language(language)

    # Lấy điểm số từ final_scores và average_scores dựa trên traits tiếng Anh
    scores = [final_scores[trait] for trait in traits_english]
    avg_scores = [average_scores[trait] for trait in traits_english]

    # Bổ sung giá trị đầu tiên vào cuối để tạo vòng tròn khép kín
    traits += [traits[0]]
    scores += [scores[0]]
    avg_scores += [avg_scores[0]]

    # Tạo radar chart với plotly
    fig = go.Figure()

    # Tạo dữ liệu hover với cả điểm và mức độ của từng trait
    # Tùy chỉnh nhãn cho điểm và mức độ theo ngôn ngữ
    score_label = "Điểm" if language == "Tiếng Việt" else "Score"
    level_label = "Mức độ" if language == "Tiếng Việt" else "Level"

    # Tạo nội dung hover với ngôn ngữ tương ứng
    hover_texts_avg = [
        f"{score_label}: {score:.2f}<br>{level_label}: {get_score_level(score, language)}" 
        for score in avg_scores
    ]
    hover_texts_user = [
        f"{score_label}: <b>{score:.2f}</b><br>{level_label}: <b>{get_score_level(score, language)}</b>" 
        for score in scores
    ]
    # Thiết lập tên dựa trên lựa chọn ngôn ngữ
    your_trait_name = "Điểm của Bạn" if language == "Tiếng Việt" else "Your Trait"
    average_scores_name = "Điểm Trung Bình" if language == "Tiếng Việt" else "Average Scores"

    # Thêm đường của Average Scores với thông tin hover
    fig.add_trace(go.Scatterpolar(
        r=avg_scores,
        theta=traits,
        fill='toself',
        name=average_scores_name,
        line=dict(color='rgba(0, 0, 255, 0.35)', dash='dashdot'),  # A lighter blue to simulate transparency
        fillcolor='rgba(204, 204, 255, 0.35)',
        hoverinfo='text',
        # hovertext=hover_texts_avg  # Hiển thị hover text
    ))


    # Thêm đường của Your Scores với thông tin hover
    fig.add_trace(go.Scatterpolar(
    r=scores,
    theta=traits,
    fill='toself',
    name=f'<span style="color:white;">{your_trait_name}</span>',
    line=dict(color='orange'),
    fillcolor='rgba(255, 165, 0, 0.25)',
    hoverinfo='text',
    hovertext=hover_texts_user,  # Display hover text
    marker=dict(
        size=9  # Increase the size of the dots
        # color='orange',  # Color of the dots
    )
))


    # Cài đặt layout được tùy chỉnh để cải thiện bố cục
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Nền của chính radar chart
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Nền của toàn biểu đồ   
        polar=dict(
            bgcolor="rgba(0, 0, 0, 0)",  # Nền trong suốt hoặc màu sắc nhẹ
            radialaxis=dict(
                visible=True,
                range=[0, 5],  # Giữ lại range từ 0-5
                showticklabels=False,  # Ẩn nhãn tick
                ticks='',
                gridcolor="#756075",  # Màu lưới nhẹ
                gridwidth=1.5,  # Độ dày lưới vừa phải
                griddash='dashdot',
                linecolor="rgba(0, 0, 0, 0)",
            ),
            angularaxis=dict(
                visible=True,
                tickfont=dict(size=12, color="rgba(150, 150, 255, 1)", family="Times"),  # Kích thước font cho nhãn
                rotation=45,  # Xoay nhãn để trông đỡ bị chồng chéo
                direction="clockwise",  # Điều chỉnh hướng nhãn
                linewidth=1.5,  # Độ dày của trục góc
                gridcolor="#756075",
                linecolor="#756075" , # Đặt màu trắng để lưới góc không quá rõ
            ),
        ),
        showlegend=True,  # Hiển thị chú thích
        hoverlabel=dict(
            bgcolor="white",  # Nền trắng
            font_size=16,  # Kích thước chữ to hơn
            font_color="black"  # Màu chữ đen
        ),
        font=dict(size=12),  # Kích thước font tổng thể
        margin=dict(l=40, r=40, b=40, t=40),  # Điều chỉnh lề cho cân đối
        # paper_bgcolor="rgba(240, 240, 240, 1)",  # Màu nền tổng thể nhẹ
        dragmode="pan"  # Cho phép kéo (xoay) nhưng tắt zoom
    )

# Thêm màu sắc và kích thước cho các traits
    colors = [ '#2774ae', '#9D03C7','#d94b24', '#ff9425','#f1d800','green'] # xanh - tim - đỏ - cam - lá - vàng
    if len(traits) > len(colors):
        # Extend the colors list by repeating it as needed
        colors = colors * (len(traits) // len(colors) + 1)
    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=[f'<b style="color:{colors[i]}; font-family:Source Sans Pro, sans-serif; font-size:17px;">{traits[i]}</b>' for i in range(len(traits))],
                tickmode='array',
            )
        ),
        transition={
        'duration': 1000,
        'easing': 'cubic-in-out'
    }
    )

    # Cập nhật hover label màu theo điểm số
    fig.update_traces(
        hoverlabel=dict(
            font=dict(size=16, color=[get_score_color(score, language) for score in scores]),
            bgcolor='white'  # Mỗi điểm có màu tương ứng với mức độ
        )
    )

    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, use_container_width=True)

# ---------------------NHẬN XÉT---------------------------------
# ____________________CHỌN NGÔN NGỮ__________________________
languages = ["Tiếng Việt", "English"]

# Thiết lập mặc định English
default_language = "Tiếng Việt"

# Cho phép người dùng chọn ngôn ngữ
language = st.sidebar.selectbox("Chọn ngôn ngữ / Language settings", languages, index=languages.index(default_language))
#----------------------CALL API-----------------------------------
# Hàm lấy nhận xét dựa trên điểm số và trait
# Đặt API key của OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
# testme = os.getenv("OPENAI_API_KEY")
# print(testme)
# client = openai(
#     api_key=os.getenv("OPENAI_API_KEY"),
# )
# Hàm gọi GPT để sinh nội dung dựa trên input
def generate_content_with_gpt(prompt, model="gpt-4o-mini", max_tokens=9000):
    try:
        # neu version new 
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7 
        )
        # Lấy nội dung phản hồi từ GPT
        print(response)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Lỗi: {e}")
        return None

# Hàm để đọc prompt từ file .txt
def load_prompt_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File {file_path} không tồn tại.")
        return None

# Hàm lưu danh sách user_hash vào file
def save_user_hash(user_hash):
    with open('txt1.txt', 'a') as file:  # Sử dụng chế độ 'a' để append
        file.write(user_hash + "\n")  # Ghi user_hash trên một dòng, thêm ký tự xuống dòng

# Hàm khôi phục danh sách user_hash từ file
def load_user_hash():
    try:
        with open('txt1.txt', 'r') as file:
            return [line.strip() for line in file.readlines()]  # Đọc từng dòng và loại bỏ khoảng trắng
    except FileNotFoundError:
        return []

def append_report_cache_to_txt(user_hash, financial_traits_text, top_traits_description, eligible_content, necessary_content):
    with open('txt3.txt', 'a') as file:  # Chế độ 'a' để thêm vào file thay vì ghi đè
        financial_traits_text = financial_traits_text.replace('\n', '\\n')
        top_traits_description = top_traits_description.replace('\n', '\\n')
        eligible_content = eligible_content.replace('\n', '\\n')

        # Kiểm tra nếu necessary_content không phải None
        if necessary_content is not None:
            necessary_content = necessary_content.replace('\n', '\\n')
        else:
            necessary_content = ""  # Nếu None, lưu chuỗi rỗng

        # Ghi dữ liệu vào file
        file.write(f"{user_hash}|{financial_traits_text}|{top_traits_description}|{eligible_content}|{necessary_content}\n")

# Hàm khôi phục report_cache từ file txt (bao gồm sản phẩm)
def load_report_cache_from_txt():
    try:
        report_cache = {}
        with open('txt3.txt', 'r') as file:
            for line in file.readlines():
                parts = line.strip().split('|', 4)  # Chỉnh thành 4 phần
                if len(parts) == 5:
                    user_hash = parts[0]
                    financial_traits_text = parts[1].replace('\\n', '\n')
                    top_traits_description = parts[2].replace('\\n', '\n')
                    eligible_content = parts[3].replace('\\n', '\n')
                    necessary_content = parts[4].replace('\\n', '\n') if parts[4] else None

                    # Lưu tất cả 4 giá trị vào cache
                    report_cache[user_hash] = (
                        financial_traits_text, 
                        top_traits_description, 
                        eligible_content, 
                        necessary_content
                    )
                else:
                    print(f"Dòng không hợp lệ: {line.strip()}")
        return report_cache
    except FileNotFoundError:
        print("File không tồn tại, khởi tạo dữ liệu mới")
        return {}


# Create a hash based on user information (birth date, time, and place)
def generate_user_hash(birth_date, birth_time, birth_place, language):
    unique_string = f"{birth_date}_{birth_time}_{birth_place}_{language}"
    return hashlib.md5(unique_string.encode()).hexdigest()

# Hàm để tính toán độ tuổi từ ngày sinh
def calculate_age(birth_date):
    today = datetime.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

# Hàm điều chỉnh giọng văn dựa trên độ tuổi
def adjust_tone_based_on_age(age):
    if age < 6:
        return "giọng văn nhẹ nhàng và dễ hiểu dành cho trẻ nhỏ", "nhỏ tuổi"
    elif 6 <= age < 19:
        return "giọng văn thân thiện và gần gũi cho lứa tuổi học sinh", "học sinh"
    elif 19 <= age < 26:
        return "giọng văn năng động và hợp thời, phù hợp với sinh viên", "sinh viên"
    elif 26 <= age < 41:
        return "giọng văn chuyên nghiệp và cụ thể, phù hợp với người đang đi làm", "đang đi làm"
    else:
        return "giọng văn trang trọng và rõ ràng, dành cho người lớn tuổi", "lớn tuổi"

def determine_score_level_and_description(trait, score):
    if 1.00 <= score <= 1.80:
        score_level = "rất thấp"
        score_description = f"Người dùng gần như không có biểu hiện mạnh mẽ trong {trait}, cho thấy xu hướng ít quan tâm đến khía cạnh này trong chi tiêu."
    elif 1.81 <= score <= 2.60:
        score_level = "thấp"
        score_description = f"Người dùng có biểu hiện yếu trong {trait}, cho thấy họ có xu hướng tránh {trait} hoặc không thường xuyên thể hiện tính cách này trong các quyết định chi tiêu."
    elif 2.61 <= score <= 3.40:
        score_level = "trung bình"
        score_description = f"Người dùng thể hiện sự cân bằng trong {trait}. Mặc dù {trait} không phải là đặc điểm nổi trội, nhưng họ có khả năng sử dụng nó trong một số quyết định chi tiêu nhất định."
    elif 3.41 <= score <= 4.20:
        score_level = "cao"
        score_description = f"Người dùng có xu hướng thể hiện {trait} thường xuyên trong các quyết định chi tiêu, cho thấy họ có xu hướng mạnh mẽ về đặc điểm này."
    else:
        score_level = "rất cao"
        score_description = f"Người dùng thể hiện {trait} một cách nổi bật, rất quyết đoán và mạnh mẽ, thường xuyên dựa vào đặc điểm này để đưa ra các quyết định chi tiêu."
    
    return score_level, score_description


# Hàm để sinh mô tả trait dựa trên GPT và độ tuổi
def get_trait_description_with_gpt(trait, score, language, age):
    # Đọc prompt từ file
    prompt_template = load_prompt_from_file('prompt_template.txt')
    
    # Kiểm tra nếu không có prompt nào
    if prompt_template is None:
        return "Không có mô tả hợp lệ."
    
    # Điều chỉnh giọng văn và nhóm tuổi dựa trên độ tuổi
    tone, age_group = adjust_tone_based_on_age(age)
    # Gọi hàm xác định mức độ điểm số và mô tả
    score_level, score_description = determine_score_level_and_description(trait, score)


    # Tạo prompt bằng cách thay thế các biến
    prompt = prompt_template.format(
        trait=trait,
        score=score,
        score_level=score_level,
        score_description=score_description,
        language=language,
        tone=tone,
        age_group=age_group
    )
    
    # Gọi GPT để sinh nội dung
    return generate_content_with_gpt(prompt)

# Hàm để sinh mô tả top 3 traits dựa trên GPT và độ tuổi
def get_top_traits_description_with_gpt(top_3_traits, final_scores, language, age):
    # Đọc prompt từ file (giả sử bạn có một file riêng cho top 3 traits)
    prompt_template = load_prompt_from_file('top_3_traits_template.txt')
    
    # Kiểm tra nếu không có prompt nào
    if prompt_template is None:
        return "Không có mô tả hợp lệ."
    
    # Điều chỉnh giọng văn và nhóm tuổi dựa trên độ tuổi
    tone, age_group = adjust_tone_based_on_age(age)
    # Gọi hàm xác định mức độ điểm số và mô tả
    # score_level, score_description = determine_score_level_and_description(trait, score)
    
    # Chuẩn bị nội dung top 3 traits để điền vào prompt
    traits_info = []
    # for trait in top_3_traits:  # Thêm vòng lặp cho top 3 traits
    for trait in final_scores:  # Lặp qua tất cả các traits trong final_scores
        score = final_scores[trait]  # Lấy điểm số của trait hiện tại từ final_scores
        score_level, score_description = determine_score_level_and_description(trait, score)
        traits_info.append(f"Trait: {trait}, Score: {score} ({score_level}) - {score_description}")

    # Tạo prompt bằng cách thay thế các biến
    prompt = prompt_template.format(top_3_traits=', '.join(top_3_traits),traits_info='\n'.join(traits_info), language=language, tone=tone, age_group=age_group)
    
    # Gọi GPT để sinh nội dung
    return generate_content_with_gpt(prompt)

# Hàm để lọc và lấy nội dung cần thiết từ phản hồi GPT (nếu cần)
def extract_content_from_gpt_response(response):
    # Giả sử bạn muốn lấy nội dung sau tiêu đề "Mô tả:"
    match = re.search(r"Mô tả:\s*(.+)", response)
    if match:
        return match.group(1)  # Lấy phần nội dung sau "Mô tả:"
    return response  # Nếu không có tiêu đề, trả về toàn bộ phản hồi

# Hàm để tính top 3 traits dựa trên final_scores
def get_top_3_traits(final_scores):
    # Sắp xếp final_scores theo giá trị (điểm) giảm dần và lấy ra 3 trait đầu tiên
    sorted_traits = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return [trait for trait, _ in sorted_traits[:3]]  # Trả về 3 traits có điểm cao nhất


# Hàm để lấy trait cao nhất và thấp nhất
def get_highest_and_lowest_trait(final_scores):
    sorted_traits = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    highest_trait = sorted_traits[0][0]  # Trait có điểm cao nhất
    lowest_trait = sorted_traits[-1][0]  # Trait có điểm thấp nhất
    return highest_trait, lowest_trait

# -----------------CALL API ĐỂ RA ĐOẠN VĂN MÔ TẢ ĐIỂM PRODUCT-----------------------------------------------------------------------
# Dictionary for product names in both languages
product_translation = {
    "goal-save": {"vi": "Mục Tiêu Cá Nhân", "en": "Goal Save"},
    "money-pot": {"vi": "Hũ Chi Tiêu", "en": "Money Pot"},
    "shared-pot":  {"vi": "Hũ Chi Tiêu Chung", "en": "Shared Pot"},
    
    "pfm": {"vi": "Quản Lý Tài Chính Cá Nhân", "en": "Personal Financial Management"},
    "spending-prediction": {"vi": "Dự Đoán Thu Chi", "en": "Spending Prediction"},
    "category-report": {"vi": "Báo Cáo Thu Chi", "en": "Cashflow Report"},
    "cashflow-overview": {"vi": "Báo Cáo Thu Chi Tổng Hợp", "en": "Cashflow Overview"},
    "asset-tracker": {"vi": "Báo Cáo Tổng Tài Sản", "en": "Asset Tracker"},
    "tags":  {"vi": "Gắn Nhãn Giao Dịch", "en": "Transaction Tags"},
    "setting-budget": {"vi": "Giới Hạn Chi Tiêu", "en": "Setting Budgets"},

    "split-bill": {"vi": "Chia Tiền Nhóm", "en": "Split Bill"},
    "schedule-payment": {"vi": "Chuyển Tiền Theo Lịch", "en": "Scheduled Payment"},
    
    "term-deposit": {"vi": "Tiết Kiệm Trực Tuyến", "en": "Term Deposit"},
    
    "overdraft": {"vi": "Thấu Chi", "en": "Overdraft"},
    "installment-loan": {"vi": "Vay Trả Góp", "en": "Installment Loan"},
    "visa-credit-card": {"vi": "Thẻ Tín Dụng VISA", "en": "VISA Credit Card"},
    
    "vinacapital-investments": {"vi": "Quỹ VCAM/VinaCapital", "en": "VCAM/VinaCapital Investments"},
    # "vcam-investment": {"vi": "Quỹ VCAM", "en": "VCAM Investment"},
    
    "present": {"vi": "Lì Xì", "en": "Lixi"},
    
    "liberty-travel-insurance": {"vi": "Bảo Hiểm Du Lịch", "en": "Travel Insurance"},
}

# Hàm lấy tên sản phẩm theo ngôn ngữ đã chọn
def get_product_name(product_key, language):
    product = product_translation.get(product_key, {})
    if language == "Tiếng Việt":
        return f"{product.get('vi')} ({product.get('en')})"
    else:
        return f"{product.get('en')} ({product.get('vi')})"
    

# Dictionary phân cấp sản phẩm
product_hierarchy = {
    "pfm": [
        "tags",
        "setting-budget",
        "category-report",
        "cashflow-overview",
        "asset-tracker",
        "spending-prediction",
    ]
}

def prepare_eligible_info(eligible_df, language, product_hierarchy):
    child_products = set(product for products in product_hierarchy.values() for product in products)
    eligible_info = ""

    # Duyệt qua từng sản phẩm trong DataFrame
    for _, row in eligible_df.iterrows():
        product = row['Product']
        product_name = get_product_name(product, language)
        label = row['Label']

        # Chọn URL dựa trên ngôn ngữ
        if language == "Tiếng Việt":
            link = f"https://timo.vn/product/{product}"
        else:
            link = f"https://timo.vn/en/{product}"

        # Kiểm tra nếu sản phẩm là sản phẩm mẹ và có sản phẩm con
        if product in product_hierarchy:
            eligible_info += f"**{product_name}** - {label}\n"
            eligible_info += f"_Bạn có thể tìm hiểu thêm tại [Link]({link})_\n"
            eligible_info += "  **Sản phẩm này bao gồm:**\n"

            # Lấy và hiển thị các sản phẩm con theo thứ tự điểm số
            child_df = eligible_df[eligible_df['Product'].isin(product_hierarchy[product])].sort_values(by='Score', ascending=False)
            for _, child_row in child_df.iterrows():
                child_name = get_product_name(child_row['Product'], language)
                child_label = child_row['Label']
                eligible_info += f"    - {child_name}: {child_label}\n"
        else:
            # Nếu không có sản phẩm con, hiển thị sản phẩm chính
            eligible_info += f"**{product_name}** - {label}\n"
            eligible_info += f"_Bạn có thể tìm hiểu thêm tại [Link]({link})_\n"

        eligible_info += "\n"  # Thêm khoảng trắng giữa các sản phẩm

    return eligible_info
    


# Lấy top 5 sản phẩm từ bảng eligible
def get_top_5_eligible_products(eligible_df):
    top_5_eligible_df = eligible_df.nlargest(5, 'Score')
    return top_5_eligible_df['Product'].tolist()

# Chuẩn bị nội dung bảng cần thiết sau khi loại bỏ các sản phẩm trùng
def prepare_necessary_info(necessary_df, eligible_df):
    # Kiểm tra nếu đầu vào là list và chuyển thành DataFrame
    if isinstance(necessary_df, list):
        necessary_df = pd.DataFrame(necessary_df, columns=['Product', 'Score', 'Label'])

    # Lấy danh sách top 5 sản phẩm từ bảng eligible
    top_5_products = get_top_5_eligible_products(eligible_df)

    # Lọc bỏ các sản phẩm đã xuất hiện trong top 5 của bảng eligible
    filtered_necessary_df = necessary_df[~necessary_df['Product'].isin(top_5_products)]

    # Nếu không còn sản phẩm cần thiết, trả về None
    if filtered_necessary_df.empty:
        return None
    
     # Chuẩn bị thông tin sản phẩm cần thiết sau khi lọc
    necessary_info = ""
    for _, row in filtered_necessary_df.iterrows():
        product = row['Product']
        product_name = get_product_name(product, language)
        label = row['Label']

        link = f"https://timo.vn/product/{product}" if language == "Tiếng Việt" else f"https://timo.vn/en/{product}"

        if language == "Tiếng Việt":
            necessary_info += f"- {product_name}: {label} _Bạn có thể tìm hiểu thêm tại [Link]({link})_\n"
        else:
            necessary_info += f"- {product_name}: {label} _Learn more at [Link]({link})_\n"


    return necessary_info

def generate_recommendation_for_eligible(eligible_df, final_scores, language, age, product_hierarchy):
    # Tạo template từ file
    prompt_template = load_prompt_from_file('product_prompt_template.txt')

    # Tập hợp tất cả các sản phẩm con
    child_products = set(product for products in product_hierarchy.values() for product in products)

    # Lấy danh sách top 10 sản phẩm theo điểm
    top_10_eligible = eligible_df.nlargest(10, 'Score')

    # Kiểm tra nếu sản phẩm mẹ nằm trong top 10
    parent_in_top_10 = set(top_10_eligible['Product']).intersection(product_hierarchy.keys())

    # Lọc ra các sản phẩm con nằm trong top 5 và có nhãn phù hợp
    child_products_in_top_5 = top_10_eligible[
        (top_10_eligible['Product'].isin(child_products)) &
        (top_10_eligible['Label'].isin([
            "Rất phù hợp", "Phù hợp", "Very Suitable", "Suitable", 
            "Cần thiết - Rất phù hợp", "Cần thiết - Phù hợp", 
            "Necessary - Very Suitable", "Necessary - Suitable"
        ]))
    ]

    # Nếu sản phẩm mẹ nằm trong top 10 và có ít nhất 3 sản phẩm con trong top 5
    if parent_in_top_10 and len(child_products_in_top_5) >= 3:
        # Loại bỏ tất cả các sản phẩm con khỏi top 5
        top_5_eligible = top_10_eligible[~top_10_eligible['Product'].isin(child_products)].nlargest(5, 'Score')
    else:
        # Lấy top 5 theo điểm nếu không cần loại bỏ sản phẩm con
        top_5_eligible = top_10_eligible.nlargest(5, 'Score')

    # Kiểm tra nếu sản phẩm mẹ nằm trong top 5
    parent_in_top_5 = set(top_5_eligible['Product']).intersection(product_hierarchy.keys())

    # Nếu sản phẩm mẹ trong top 5, liệt kê sản phẩm con dưới mẹ
    eligible_info = ""
    for _, row in top_5_eligible.iterrows():
        product = row['Product']
        product_name = get_product_name(product, language)
        label = row['Label']
        link = f"https://timo.vn/product/{product}" if language == "Tiếng Việt" else f"https://timo.vn/en/{product}"

        eligible_info += f"**{product_name}** - {label}\n"
        eligible_info += f"_Bạn có thể tìm hiểu thêm tại [Link]({link})_\n"

        # Nếu sản phẩm là mẹ, liệt kê các sản phẩm con theo điểm
        if product in product_hierarchy:
            eligible_info += "  **Sản phẩm này bao gồm:**\n"
            child_df = eligible_df[
                (eligible_df['Product'].isin(product_hierarchy[product])) &
                (eligible_df['Label'].isin(["Rất phù hợp", "Phù hợp", "Very Suitable", "Suitable"]))
            ].sort_values(by='Score', ascending=False)

            for _, child_row in child_df.iterrows():
                child_name = get_product_name(child_row['Product'], language)
                child_label = child_row['Label']
                eligible_info += f"    - {child_name}: {child_label}\n"

        eligible_info += "\n"  # Thêm khoảng trắng giữa các sản phẩm

    # Chuẩn bị danh sách top 5 sản phẩm để hiển thị
    top_5_eligible_info = "\n".join([
        f"{i + 1}. {get_product_name(row['Product'], language)} - {row['Label']}"
        for i, row in top_5_eligible.iterrows()
    ])

    # Chuẩn bị thông tin về traits
    traits_info = []
    for trait, score in final_scores.items():
        result = determine_score_level_and_description(trait, score)
        if isinstance(result, tuple) and len(result) == 2:
            level, description = result
            traits_info.append(f"Trait: {trait}, Score: {score} ({level}) - {description}")
        else:
            raise ValueError(f"Unexpected return from determine_score_level_and_description: {result}")

    # Điều chỉnh tone và độ tuổi người dùng
    tone, age_group = adjust_tone_based_on_age(age)

    # Format nội dung prompt với template
    prompt = prompt_template.format(
        traits_info='\n'.join(traits_info),
        top_5_eligible_products=top_5_eligible_info,
        eligible_products=eligible_info,
        language=language,
        tone=tone,
        age_group=age_group
    )

    # Gọi hàm GPT để sinh nội dung từ prompt
    return generate_content_with_gpt(prompt)




# Tạo nội dung cho sản phẩm cần thiết nếu có
def generate_recommendation_for_necessary(necessary_df, eligible_df, final_scores, language, age):
    # Chuẩn bị thông tin cần thiết sau khi lọc
    necessary_info = prepare_necessary_info(necessary_df, eligible_df)

    # Kiểm tra nếu không còn sản phẩm cần thiết
    if necessary_info is None:
        return None  # Không cần gọi API GPT

    prompt_template = load_prompt_from_file('necessary_product_prompt_template.txt')
    tone, age_group = adjust_tone_based_on_age(age)

    # Chuẩn bị nội dung top 3 traits để điền vào prompt
    traits_info = []
    for trait, score in final_scores.items():
        result = determine_score_level_and_description(trait, score)  # Hàm phải trả về tuple (level, description)

        if isinstance(result, tuple) and len(result) == 2:
            level, description = result
            traits_info.append(f"Trait: {trait}, Score: {score} ({level}) - {description}")
        else:
            raise ValueError(f"Unexpected return from determine_score_level_and_description: {result}")

    prompt = prompt_template.format(
        traits_info='\n'.join(traits_info),
        necessary_products=necessary_info,
        language=language,
        tone=tone,
        age_group=age_group
    )
    print(generate_recommendation_for_eligible(eligible_df, final_scores, "Tiếng Việt", 30, product_hierarchy))
    return generate_content_with_gpt(prompt)



# -----------------HỆ THỐNG ĐIỂM PRODUCT-----------------------------------------------------------------------

# Hàm để đọc điều kiện từ file CSV và chuyển thành hàm lambda
def load_conditions_from_csv(file_path):
    conditions_df = pd.read_csv(file_path)
    conditions_dict = {}
    for _, row in conditions_df.iterrows():
        # Sử dụng `eval` để chuyển chuỗi điều kiện thành hàm lambda thực thi được
        conditions_dict[row['Product']] = eval(f"lambda scores: {row['Condition']}")
    return conditions_dict

# Đọc product_conditions từ file CSV
product_conditions = load_conditions_from_csv('product_conditions.csv')

# Đọc necessity_rules từ file CSV
necessity_rules = load_conditions_from_csv('necessity_rules.csv')


def load_trait_weights_from_csv(file_path):
    # Đọc file CSV và chỉ lấy 7 cột đầu tiên
    trait_weights_df = pd.read_csv(file_path, usecols=range(7))
    
    # Chuyển DataFrame thành dictionary, với sản phẩm làm key và trọng số làm value
    trait_weights_dict = {}
    for _, row in trait_weights_df.iterrows():
        product = row['Product']
        trait_weights = row.drop('Product').to_dict()
        trait_weights_dict[product] = trait_weights
    return trait_weights_dict

# Reload trait weights from the provided CSV file
trait_weights_file_path = 'trait_weights_products.csv'
trait_weights_dict = load_trait_weights_from_csv(trait_weights_file_path)

# RandomForest function for model-based eligibility checking
def evaluate_product_with_model(clf, final_scores):
    features = np.array(list(final_scores.values())).reshape(1, -1)
    prediction = clf.predict(features)
    return prediction[0]

# Check eligibility for products
def check_product_eligibility(product, final_scores, clf=None):
    if clf:
        return evaluate_product_with_model(clf, final_scores)
    if product in product_conditions:
        return product_conditions[product](final_scores)
    return True

# Check necessity for products
def evaluate_product_necessity(final_scores, product, clf=None):
    if clf:
        return evaluate_product_with_model(clf, final_scores)
    if product in necessity_rules:
        return necessity_rules[product](final_scores)
    return False

def calculate_product_scores_numpy(final_scores, product_keywords, keyword_to_trait_mapping, trait_weights_dict, weight_factor=1.0):
    trait_names = list(final_scores.keys())  # Các đặc điểm trong final_scores
    trait_scores = np.array([final_scores[trait] for trait in trait_names])  # Điểm số cho từng đặc điểm

    product_scores = {}
    for product, keywords in product_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in keyword_to_trait_mapping:
                # Lấy trọng số của keyword_trait từ keyword_to_trait_mapping
                keyword_trait_weights = np.array([keyword_to_trait_mapping[keyword].get(trait, 0) for trait in trait_names])
                print(f"Keyword: {keyword}, Trait Weights from Keyword: {keyword_trait_weights}")

                # Lấy trọng số đặc điểm từ trait_weights_dict
                if product in trait_weights_dict:
                    product_trait_weights = np.array([trait_weights_dict[product].get(trait, 1) for trait in trait_names])
                else:
                    product_trait_weights = np.ones_like(keyword_trait_weights)  # Mặc định nếu không có trọng số
                print(f"Trait Weights from Product: {product_trait_weights}")
                print(f"Trait Scores: {trait_scores}")

                # Kiểm tra trước khi cộng dồn
                print(f"Score before adding keyword {keyword}: {score}")
                
                # Tính điểm theo công thức mới
                score_contribution =  weight_factor * np.sum(trait_scores * product_trait_weights * keyword_trait_weights)
                print(f"Score contribution from keyword {keyword}: {score_contribution}")
                
                # Cộng dồn
                score += score_contribution
                print(f"Score after adding keyword {keyword}: {score}")
                
        print(f"Total score for product {product}: {score}")
        product_scores[product] = {
            'Score': score,
            'Eligible': check_product_eligibility(product, final_scores),
            'Necessary': evaluate_product_necessity(final_scores, product)
        }
    print(product_scores)
    return product_scores



# def assign_labels_using_kmeans(product_scores, max_clusters=10, random_state=42):
# def assign_labels_using_normalize(product_scores, language=language):
#     labeled_scores = {}

#     # Lấy điểm số từ các sản phẩm
#     scores = np.array([result.get('Score', 0) for result in product_scores.values()])

#     # Tính phân vị (percentiles) dựa trên điểm số
#     percentiles = np.percentile(scores, [30, 60, 90])  # Lấy các phân vị 25%, 50%, và 75%


#     # Lặp qua từng sản phẩm và gán nhãn dựa trên phân vị
#     for i, (product, result) in enumerate(product_scores.items()):
#         score = result.get('Score', 0)  # Đảm bảo 'Score' luôn tồn tại
#         eligible = result.get('Eligible', True)  # Giá trị mặc định là True
#         necessary = result.get('Necessary', False)  # Giá trị mặc định là False

#         # Gán nhãn dựa trên giá trị của ngôn ngữ
#         if language == "Tiếng Việt":
#             label_names = ["Rất phù hợp", "Phù hợp", "Ít quan tâm", "Có thể không quan tâm"]
#         else:
#             label_names = ["Very Suitable", "Suitable", "Little interest", "Might not be interested"]

#         # Nếu sản phẩm không hợp lệ, gán nhãn luôn là "Có thể không quan tâm"
#         if not eligible:
#             label_name = label_names[-1]  # "Có thể không quan tâm" hoặc "Might not be interested"
#         else:
#             # Gán nhãn dựa trên phân vị
#             if score >= percentiles[2]:  # Trên phân vị 75%
#                 label_name = label_names[0]  # "Rất phù hợp" hoặc "Very Suitable"
#             elif percentiles[1] <= score < percentiles[2]:  # Từ phân vị 50% đến 75%
#                 label_name = label_names[1]  # "Phù hợp" hoặc "Suitable"
#             elif percentiles[0] <= score < percentiles[1]:  # Từ phân vị 25% đến 50%
#                 label_name = label_names[2]  # "Ít quan tâm" hoặc "Little interest"
#             else: # Dưới phân vị 25%
#                 label_name = label_names[3]  # "Có thể không quan tâm" hoặc "Might not be interested"

#             # Điều chỉnh nhãn nếu sản phẩm là "Necessary"
#         if necessary:
#             label_name = f"Necessary - {label_name}" if language == "English" else f"Cần thiết - {label_name}"

#         # Lưu thông tin sản phẩm cùng nhãn
#         labeled_scores[product] = {'Score': score, 'Eligible': eligible, 'Necessary': necessary, 'Label': label_name}

#     return labeled_scores
def assign_labels_using_normalize(product_scores, language=language):
    labeled_scores = {}

    # Tạo DataFrame để sắp xếp các sản phẩm theo điểm
    df = pd.DataFrame.from_dict(product_scores, orient='index')
    df['Rank'] = df['Score'].rank(ascending=False, method='first')

    # Tính số lượng sản phẩm
    total_products = len(df)

    # Tính các ngưỡng để phân nhóm (25% top, 50% middle, 25% bottom)
    top_threshold = total_products * 0.30
    mid_threshold = total_products * 0.60
    bottom_threshold = total_products * 0.90

        # Gán nhãn dựa trên giá trị của ngôn ngữ
    if language == "Tiếng Việt":
        label_names = ["Rất phù hợp", "Phù hợp", "Ít quan tâm", "Có thể không quan tâm"]
    else:
        label_names = ["Very Suitable", "Suitable", "Little interest", "Might not be interested"]

    # Gán nhãn cho từng sản phẩm dựa trên rank
    for product, result in product_scores.items():
        rank = df.loc[product, 'Rank']
        score = result.get('Score', 0)
        eligible = result.get('Eligible', True)
        necessary = result.get('Necessary', False)

        if rank <= top_threshold:
            label_name = label_names[0]  # "Rất phù hợp"
        elif rank <= mid_threshold:
            label_name = label_names[1]  # "Phù hợp"
        elif rank <= bottom_threshold:
            label_name = label_names[2]  # "Phù hợp"    
        else:
            label_name = label_names[3]  # "Không phù hợp"

        if necessary:
            label_name = f"Necessary - {label_name}" if language == "English" else f"Cần thiết - {label_name}"

        # Lưu thông tin sản phẩm cùng nhãn
        labeled_scores[product] = {
            'Score': score,
            'Rank': rank,
            'Eligible': eligible,
            'Necessary': necessary,
            'Label': label_name
        }

    return labeled_scores

# Get final product scores (combines eligibility, necessity, and scoring)
def get_final_product_scores(final_scores, product_keywords, keyword_to_trait_mapping, clf=None, language="Tiếng Việt"):
    product_scores = calculate_product_scores_numpy(final_scores, product_keywords, keyword_to_trait_mapping)

    product_info = {}
    for product, score in product_scores.items():
        is_eligible = check_product_eligibility(product, final_scores, clf)
        is_necessary = evaluate_product_necessity(final_scores, product, clf)
        
        product_info[product] = {
            'Score': score,
            'Eligible': is_eligible,
            'Necessary': is_necessary
        }

    # Điều chỉnh gán nhãn theo ngôn ngữ
    product_info = assign_labels_using_normalize(product_info, language=language)
    return product_info
#--------------------------------RATE APP-----------------------------------------------

# Hàm lưu đánh giá vào file JSON
def save_feedback(stars, comment, language):
    feedback_data = {
        "rating": stars,
        "comment": comment,
        "language": language,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Tạo file nếu chưa có, hoặc ghi vào file có sẵn
    try:
        with open("feedback.json", "r+") as file:
            data = json.load(file)
            data.append(feedback_data)
            file.seek(0)
            json.dump(data, file, indent=4)
    except FileNotFoundError:
        with open("feedback.json", "w") as file:
            json.dump([feedback_data], file, indent=4)

# Lưu trạng thái khi nhấn nút Calculate bằng session state
if "calculate_pressed" not in st.session_state:
    st.session_state.calculate_pressed = False

# ----------------------------Streamlit UI---------------------------------------------

# Streamlit UI
# if language == "Tiếng Việt":
#     banner_url = "https://timo.vn/wp-content/uploads/2024/10/AstroTomi_Email-banner_VN.jpg"
# else:
#     banner_url = "https://timo.vn/wp-content/uploads/2024/10/AstroTomi_Email-banner_ENG.jpg"

# # CSS để điều chỉnh khoảng cách giữa banner và nội dung
# st.markdown(
#     f"""
#     <div style="text-align: center; margin-bottom: 50px;">
#         <img src="{banner_url}" 
#              alt="AstroTomi Header" style="width:100%; max-width:700px; object-fit: cover;">
#     </div>
#     """,
#     unsafe_allow_html=True
# )

st.markdown(
    """
    <style>
    .emoji {
        font-size: 50px;
        text-align: center;
    }
    .title {
        font-size: 50px;
        color: #6A0DAD;
        text-align: center;
        display: inline-block;
        background: linear-gradient(90deg, rgba(106,13,173,1) 0%, rgba(163,43,237,1) 50%, rgba(252,176,69,1) 100%);
        -webkit-background-clip: text;
        color: transparent;
        margin: 0 10px;
        margin-bottom: -40px; /* Giảm khoảng cách giữa tiêu đề và expander */
    }
    .expander {
        margin-top: -30px; /* Điều chỉnh khoảng cách phía trên của expander */
    }
    .beta {
        font-size: 20px;
        # color: #6A0DAD;
        text-align: center;
        margin-top: -10px;
    }
    .container {
        display: flex;
        justify-content: center;
        align-items: center;
        # margin-bottom: 80px; /* Điều chỉnh khoảng cách giữa tiêu đề và tab */
    }

    /* Điều chỉnh khoảng cách tab */
    .stTabs {
        margin-top: -60px; /* Điều chỉnh khoảng cách giữa tiêu đề và tab */
    
    }
    </style>
    <div class="container">
        <div class="title">Hello nunu</div>
        <!--<div class="emoji">✨</div>
        <div class="title">ASTROTOMI</div>
        <div class="beta">(Beta)</div>
        <div class="emoji">✨</div>-->
    </div>
    """,
    unsafe_allow_html=True
)
# Mã hóa hình ảnh 'tomi.png' thành chuỗi base64
# def encode_image(image_path):
#     with open(image_path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode()

# # Đường dẫn đến ảnh 'tomi.png'
# tomi_path = os.path.join("images", "Tomi-Sticker_Part1_18.png")

# # Mã hóa ảnh
# encoded_tomi = encode_image(tomi_path)

# # CSS tùy chỉnh để hiển thị ảnh ở góc phải màn hình
# st.markdown(
#     f"""
#     <style>
#     .tomi-image {{
#         position: fixed;
#         top: 100px;  /* Điều chỉnh khoảng cách từ trên xuống */
#         right: 200px;  /* Căn hình về phía bên phải */
#         width: 150px;  /* Đặt kích thước ảnh */
#         height: auto;
#         z-index: 10;  /* Đảm bảo ảnh nằm trên các phần tử khác */
#     }}
#     </style>
#     <img src="data:image/png;base64,{encoded_tomi}" class="tomi-image"/>
#     """,
#     unsafe_allow_html=True
# )
st.markdown(
    """
    <style>

     /* Tạo lớp phủ nền riêng biệt */
    .bg-container {
        position: fixed;
        bottom: -170px;  /* Đặt ở cạnh dưới của màn hình */
        left: -100px;  /* Căn về góc trái */
        width: 800px;  /* Tăng kích thước ảnh lên */
        height: 800px;  /* Giữ tỷ lệ phù hợp */
        # background-image: url('https://chiemtinhlaso.com/assets/images/hand_bg.png');
        background-size: contain;  /* Giữ nguyên tỷ lệ ảnh */
        background-position: left bottom;  /* Đặt ở góc trái dưới */
        background-repeat: no-repeat;  /* Không lặp lại ảnh */
        opacity: 0.7;  /* Độ trong suốt */
        pointer-events: none;  /* Ngăn tương tác của người dùng với ảnh nền */
        animation: rotate-bg 3600s infinite linear;  /* Xoay ảnh liên tục */
    }

    /* Animation xoay vòng cho background */
    @keyframes rotate-bg {
        from {
            transform: rotate(0deg);
        }
        to {
            transform: rotate(360deg);
        }
    }

    /* Đảm bảo phần nội dung không bị ảnh hưởng */
    [data-testid="stApp"] > div {
        position: relative;
        overflow-y: auto;  /* Cho phép cuộn nội dung */
        height: 100%;  /* Đảm bảo nội dung sử dụng toàn bộ chiều cao */
    }

    /* Đổi nền của sidebar bằng ảnh */
    [data-testid="stSidebar"] {
        # background-image: url('https://images.newscientist.com/wp-content/uploads/2023/07/03093742/sei162306009.jpg');
        background-size: cover;  /* Ảnh sẽ bao phủ toàn bộ sidebar */
        background-repeat: no-repeat;
        background-position: relative;
        resize: horizontal;  /* Cho phép kéo ngang */
        # overflow: auto;  /* Đảm bảo nội dung có thể cuộn nếu vượt quá kích thước */
    }

    /* Tạo lớp overlay bán trong suốt */
    [data-testid="stSidebar"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        # background-color: rgba(0, 0, 0, 0.15); /* Điều chỉnh độ trong suốt tại đây */
        z-index: 0;  /* Đặt lớp overlay phía sau nội dung */
    }

    /* Lật ngược chỉ phần background */
    [data-testid="stSidebar"]::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: inherit;  /* Kế thừa hình ảnh nền */
        background-size: cover;
        background-position: center; #relative;
        transform: scaleY(-1);  /* Lật ngược hình nền */
        z-index: -1;  /* Đặt hình nền phía sau overlay */
        opacity: 0.7;  /* Đặt độ mờ nếu cần */
    }

    /* Đổi màu chữ của tiêu đề trong sidebar */
    [data-testid="stSidebar"] label {
        color: #f8e4ff;
    }

    [data-testid="stSidebar"] h2 {
        color: #f6d1fb; /* h2 là nhập thông tin... */
    }

    /* Đổi màu chữ cho các phần tử text, button */
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-1l02zno, [data-testid="stSidebar"] .css-1vbd788, [data-testid="stSidebar"] .css-1f97m6v {
        color: #0e0834 !important;
    }

    /* Đổi màu cho nút button trong sidebar */
    [data-testid="stSidebar"] button {
        background-color: #601b9f;
        color: white;
    }
 
    /* Đổi màu cho radio buttons */
    [data-testid="stSidebar"] .stRadio label {
        color: #f6d1fb !important;  /*  Màu chữ của AM/PM  */
    }

    /* Đổi màu của placeholder trong input text */
    [data-testid="stSidebar"] input::placeholder {
        color: #0e0834 !important;  /*input text*/
    }
    
    /* Đổi màu chữ cho label của date_input trong sidebar */
    [data-testid="stSidebar"] div.stDateInput label {
        color: #f6d1fb !important;  /* yymmdd */
    }

    /* Đổi màu chữ cho label của number_input trong sidebar */
    [data-testid="stSidebar"] div.stNumberInput label {
        color: #f6d1fb !important;  /* hour minute */ #630c80
    }

    /* Đổi màu chữ trong input */
    input {
        color: f6d1fb !important;
    }

    [data-testid="stSidebar"] {
        width: 400px;  /* Điều chỉnh chiều rộng của sidebar */
    }

    /* Thay đổi màu nền của expander */
    [data-testid="stExpander"] {
        # background-color: rgba(240, 242, 246, 0.5); /* Thay đổi thành màu bạn muốn */
        border: 1px solid #8A2BE2 !important;  /* Màu tím cho viền */
        border-radius: 10px;  /* Bo tròn viền */
        margin-bottom: 15px;
    }
    [data-testid="stExpander"] .streamlit-expanderHeader {
        color: #8A2BE2 !important;  /* Màu tím cho tiêu đề */
        font-weight: bold;
        font-size: 28px !important; 
    }
    
    /* Thay đổi màu nền của table */
    .custom-table {
        # background-color: rgba(240, 242, 246, 0.5);  /* Màu nền của bảng */
        # color: black;  /* Màu chữ cho tiêu đề cột */
        border-collapse: collapse;
        width: 99%;
        text-align: left;
    }
    .custom-table th {
        # background-color: rgba(98, 102, 166, 0.1);  /* Màu nền cho tiêu đề cột */
        # color: #7f6d84;  /* Màu chữ cho tiêu đề cột */
        text-align: left;
        padding: 4px;
    }
    .custom-table td {
        # background-color: rgba(98, 102, 166, 0.1);  /* Màu nền cho các hàng */
        # color: black;  /* Màu chữ cho các hàng */
        padding: 4px;
    }
    # .custom-table tr:nth-child(even) {
    #     background-color: rgba(98, 102, 166, 0.1);  /* Màu nền cho hàng chẵn */
    # }

    </style>
    # <div class="bg-container"></div>
    """,
    unsafe_allow_html=True
)

with st.expander('Nhập thông tin của bạn:' if language == "Tiếng Việt" else "Enter your information:", expanded=True):
   
        # Dịch văn bản dựa trên ngôn ngữ đã chọn
        if language == "Tiếng Việt":
            date_label = 'Ngày sinh (yyyy/mm/dd)'
            hour_label = "Giờ"
            minute_label = "Phút"
            am_pm_label = "AM/PM"
            not_sure_time_msg = "Nếu không rõ giờ sinh, hãy để 11h59p AM."
            birth_place_label = "Nhập tên thành phố:"
            suggestion_label = "Chọn một thành phố từ gợi ý:"
            no_suggestions_msg = "Hệ Thống đang quả tải vì lượng truy cập cao, vui lòng thử lại sau ít phút nữa."
            enter_city_msg = "Nhập tên thành phố để tìm kiếm nơi sinh của bạn."
            text = "Vui lòng nhập ngày tháng năm sinh mà bạn được sinh ra"
            calculate_button_label = "Let's go" 
            refresh_button_label = "Create new" 
        else:
            date_label = 'Date of Birth (yyyy/mm/dd)'
            hour_label = "Hour"
            minute_label = "Minute"
            am_pm_label = "AM/PM"
            not_sure_time_msg = "If you're unsure of the birth time, use 11:59 AM."
            birth_place_label = "Enter city name:"
            suggestion_label = "Select a city from suggestions:"
            no_suggestions_msg = "Please Try Again Later."
            enter_city_msg = "Enter city or country name to see suggestions."
            text = "Please enter the date you were born"
            calculate_button_label = "Calculate"  
            refresh_button_label = "Refresh" 

        # Nhập ngày sinh
        birth_date = st.date_input(date_label, min_value=datetime(1700, 1, 1), max_value=datetime.today())
        age = calculate_age(birth_date)

        st.markdown(
            f'<p style="color:white;">{text}</p>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            hour = st.number_input(hour_label, min_value=0, max_value=12, value=11)  # Đặt giá trị mặc định là 11
        with col2:
            minute = st.number_input(minute_label, min_value=0, max_value=59, value=59)  # Đặt giá trị mặc định là 59
        with col3:
            am_pm = st.radio(am_pm_label, ["AM", "PM"], index=0)  # Chọn AM làm mặc định

        # Thêm câu thông báo về giờ sinh không rõ
        st.markdown(
            f'<p style="color:white;">{not_sure_time_msg}</p>',
            unsafe_allow_html=True
        )

        # Chuyển đổi sang định dạng 24 giờ
        if am_pm == "PM" and hour != 12:
            hour += 12
        elif am_pm == "AM" and hour == 12:
            hour = 0

        # Nhập địa điểm sinh
        birth_place = st.text_input(birth_place_label)
        if birth_place:
            suggestions = get_city_suggestions(birth_place)
            if suggestions:
                selected_city = st.selectbox(suggestion_label, suggestions)
                st.markdown(
                    f'<p style="color:white;">{f"You have selected: {selected_city}" if language == "English" else f"Bạn đã chọn: {selected_city}"}</p>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<p style="color:white;">{no_suggestions_msg}</p>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                f'<p style="color:white;">{enter_city_msg}</p>',
                unsafe_allow_html=True
            )
# Thêm CSS để tạo khoảng cách giữa nút Calculate và Tabs
st.markdown(
    """
    <style>
    .button-container {
        display: flex;
        justify-content: space-between; /* Đặt nút Calculate bên trái và Refresh bên phải */
        align-items: center;
    }
    .calculate-button {
        margin-right: 10px;
    }
    .stTabs {
        margin-top: 20px; /* Tạo khoảng cách trên các tab */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Thêm nút "Calculate" và "Refresh" trong cùng một hàng, với ngôn ngữ động

    # Khi nhấn nút "Calculate", ẩn form và hiển thị kết quả
if st.button(f"✨ {calculate_button_label} ✨"):
            if not birth_place.strip():
                # Hiển thị thông báo lỗi nếu chưa nhập địa điểm sinh
                if language == "Tiếng Việt":
                    st.error("Bạn chưa nhập nơi sinh.")
                else:
                    st.error("You haven't input your birth place.")
            else:
                lat, lon, timezone = get_location_and_timezone(birth_place)
                positions = get_planet_positions(birth_date.year, birth_date.month, birth_date.day, hour, minute, lat, lon, timezone)
                ascendant, houses = calculate_houses_and_ascendant(birth_date.year, birth_date.month, birth_date.day, hour, minute, lat, lon, timezone)

                # Tạo danh sách individual_planets từ kết quả positions
                individual_planets = [(planet, get_zodiac_sign_and_degree(degree)[0]) for planet, degree in positions.items()]

                # Hiển thị bảng vị trí các hành tinh với House Cup và House Ruler (bao gồm cả Ascendant)
                df_positions = create_astrology_dataframe(positions, houses, ascendant)
                df_houses = create_house_dataframe(houses, ascendant)

                # Tính toán và hiển thị bảng các góc aspect
                aspects = calculate_aspects(positions)
                aspect_degree_mapping = {
                'Conjunction': 0,
                'Sextile': 60,
                'Square': 90,
                'Trine': 120,
                'Quincunx': 150,
                'Opposition': 180
                }


                # Tạo DataFrame cho các góc hợp (aspects)
                df_aspects = create_aspects_dataframe(aspects, positions)


                # Thay đổi cột Degree bằng cách ánh xạ Aspect
                df_aspects['Degree'] = df_aspects['Aspect'].map(aspect_degree_mapping)
                df_aspects['Degree'] = df_aspects['Degree'].apply(lambda x: f"{x}°")
                # Hiển thị bảng đã cập nhật với các cột theo thứ tự mong muốn
                df_aspects = df_aspects[['Planet 1', 'Zodiac Sign 1', 'Aspect', 'Degree', 'Planet 2', 'Zodiac Sign 2']]



                # Tính toán các đặc điểm tài chính
                formatted_aspects = format_aspects({'Aspects': str(aspects)}, individual_planets)
                relevant_planets = [planet for planet, sign in individual_planets]  # Chỉ lấy tên các hành tinh
                extracted_aspects = extract_relevant_aspects(formatted_aspects, relevant_planets)

                final_scores = calculate_financial_traits(individual_planets, formatted_aspects)

                average_scores = {trait: 3.0 for trait in final_scores.keys()}  # Ví dụ tất cả trung bình là 3.0

                final_product_scores = calculate_product_scores_numpy(final_scores, product_keywords, keyword_to_trait_mapping, trait_weights_dict)

                # Sử dụng KMeans clustering để gán nhãn
                labeled_product_scores = assign_labels_using_normalize(final_product_scores)

                # Nhãn cho các trạng thái sản phẩm
                label_names = {
                    "Tiếng Việt": {
                        "Very Suitable": "Rất phù hợp",
                        "Suitable": "Phù hợp",
                        "Little interest": "Ít quan tâm",
                        "Might not be interested": "Có thể không quan tâm"
                    },
                    "English": {
                        "Very Suitable": "Very Suitable",
                        "Suitable": "Suitable",
                        "Little interest": "Little interest",
                        "Might not be interested": "Might not be interested"
                    }
                }

                # Chọn ngôn ngữ hiện tại
                current_language = language  # Hoặc "Tiếng Việt" hoặc "English"

                # Phân loại sản phẩm theo Eligible và Necessary
                eligible_products = [
                    (product, result['Score'], label_names[current_language].get(result['Label'], result['Label']))
                    # (product, result['Score'], result['Label'].replace("Cần thiết - ", "").replace("Necessary - ", ""))
                    for product, result in labeled_product_scores.items() 
                ]

                necessary_products = [
                    (product, result['Score'], label_names[current_language].get(result['Label'], result['Label']))
                    for product, result in labeled_product_scores.items() if result['Necessary']
                ]


                # Tạo DataFrame cho Eligible và Necessary Products
                eligible_df = pd.DataFrame(eligible_products, columns=['Product', 'Score', 'Label'])
                necessary_df = pd.DataFrame(necessary_products, columns=['Product', 'Score', 'Label'])

                # Mức độ ưu tiên của các nhãn
                priority = {
                    "Rất phù hợp": 1,
                    "Phù hợp": 2,
                    "Ít quan tâm": 3,
                    "Có thể không quan tâm": 4
                } if current_language == "Tiếng Việt" else {
                    "Very Suitable": 1,
                    "Suitable": 2,
                    "Little interest": 3,
                    "Might not be interested": 4
                }

                def custom_sort(row):
                    label = row['Label']
                    if current_language == "Tiếng Việt":
                        if "Cần thiết" in label:
                            label_without_necessity = label.replace("Cần thiết - ", "")
                            return (2, priority.get(label_without_necessity, 5))  
                        else:
                            return (1, priority.get(label, 5))
                    elif current_language == "English":
                        if "Necessary" in label:
                            label_without_necessity = label.replace("Necessary - ", "")
                            return (2, priority.get(label_without_necessity, 5))
                        else:
                            return (1, priority.get(label, 5))


                # Sắp xếp theo thứ tự ưu tiên và hiển thị cho các sản phẩm
                eligible_df['Priority'] = eligible_df['Label'].map(priority)  
                eligible_df = eligible_df.sort_values(by=['Priority', 'Score'], ascending=[True, False]).drop(columns='Priority').reset_index(drop=True)

                necessary_df['Sort_Order'] = necessary_df.apply(custom_sort, axis=1)
                necessary_df = necessary_df.sort_values(by=['Sort_Order', 'Score', 'Label'], ascending=[True, False, True]).drop(columns=['Sort_Order']).reset_index(drop=True)

                # Tạo DataFrame cho tất cả các sản phẩm


                # Sắp xếp và hiển thị cho tất cả các sản phẩm
                all_products = [
                    (product, result['Score'], label_names[current_language].get(result['Label'], result['Label']))
                    for product, result in labeled_product_scores.items()
                ]
                all_products_df = pd.DataFrame(all_products, columns=['Product', 'Score', 'Label'])
                all_products_df['Priority'] = all_products_df['Label'].apply(lambda x: 0 if x.startswith("Necessary - ") or x.startswith("Cần thiết - ") else 1)

                all_products_df = all_products_df.sort_values(by=['Priority', 'Score'], ascending=[True, False]).drop(columns='Priority').reset_index(drop=True)

            # Tạo các tab cho các phần khác nhau
            # tab1, tab2, tab3, tab4 = st.tabs(["Astrology", "Financial Traits", "Product Recommendations", "Rating"])
            # Thay đổi tên các tab theo ngôn ngữ
            if language == "Tiếng Việt":
                tab_titles = ["Tính Cách Tài Chính", "✨Đánh giá✨"]
                # rating_label = "Đánh giá ứng dụng từ 1 đến 5 sao"
                # comment_label = "Bình luận về ứng dụng"
                feedback_message = "### Hãy giúp Tomi chọn mức độ hài lòng mà bạn cảm thấy khi trải nghiệm bản beta của ASTROTOMI nha!"
            else:
                tab_titles = ["Financial Traits", "✨Feedback✨"]
                # rating_label = "Rate the app from 1 to 5 stars"
                # comment_label = "Comment on the app"
                feedback_message = "### Please help Tomi select how you feel about the ASTROTOMI beta experience!"

            # Đường dẫn đến các ảnh sticker đã lưu cục bộ
            # sticker_1_path = os.path.join("images", "Tomi-Sticker_Part1_18.png")
            # sticker_2_path = os.path.join("images", "Tomi-Sticker_Part1_31.png")
            # sticker_3_path = os.path.join("images", "Tomi-Sticker_Part1_36.png")
            # Văn bản cho các sticker theo ngôn ngữ
            texts = {
                "Tiếng Việt": ["Chán òm", "Hơi khó hiểu~~", "Tuyệt vời!!"],
                "English": ["Not Interested", "So-So~~", "Amazing!"]
            }

            def display_sticker(image_path, link, width=100, text=""):
                """Hàm hiển thị sticker dưới dạng nút bấm với liên kết và văn bản."""
                if os.path.exists(image_path):
                    with open(image_path, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()

                    st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <a href="{link}" target="_blank">
                                <img class="sticker-button" src="data:image/png;base64,{img_data}" width="{width}" style="cursor: pointer;"/>
                            </a>
                            <p style="margin-top: 5px; font-size: 18px;">{text}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"Ảnh không tồn tại: {image_path}")

            # Tạo tab với tên theo ngôn ngữ đã chọn
            tabs = st.tabs(tab_titles)
            
            # Hàm chuyển Markdown sang HTML nếu cần thiết
            def markdown_to_html(markdown_text):
                return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', markdown_text)


            # Tab 1: Hiển thị vị trí các hành tinh
            with tabs[0]:
                
            #         st.write("### Planetary Positions:")
            #         st.dataframe(df_positions)
            #         st.write("### House Information:")
            #         st.dataframe(df_houses)
            #         st.write("### Planetary Aspects:")
            #         st.dataframe(df_aspects)
                    

            # Tab 2: Hiển thị Radar Chart cho các đặc điểm tài chính
                if 'report_cache' not in st.session_state:
                    st.session_state.report_cache = load_report_cache_from_txt()
                if 'user_hash' not in st.session_state:
                    st.session_state.user_hash = load_user_hash()

            # with tabs[1]:
                st.write("                         ")
                if not birth_place.strip():
                    # Nếu thiếu nơi sinh, hiển thị thông báo lỗi và dừng các thao tác khác
                    if language == "Tiếng Việt":
                        st.error("Bạn vui lòng điền đầy đủ thông tin nha.")
                    else:
                        st.error("You haven't input your birth place.")
                else:
                    # Nếu đã nhập đầy đủ thông tin, tiếp tục hiển thị nội dung
                    if language == "Tiếng Việt":
                        st.write("### Biểu đồ dựa trên hành vi tài chính của bạn:")
                    else:
                        st.write("### Financial Traits Radar Chart:")
                                            
                    plot_radar_chart(final_scores, average_scores, language)

                    # Từ điển dịch các traits từ tiếng Anh sang tiếng Việt
                    trait_translation = {
                        "Adventurous": "Mạo Hiểm",
                        "Convenient": "Tiện Lợi",
                        "Impulsive": "Phóng Khoáng",
                        "Conservative": "Kiên Định",
                        "Analytical": "Tỉ Mỉ",
                        "Cautious": "Cẩn Trọng"
                    }

                    # Hàm dịch tên trait theo ngôn ngữ
                    def get_translated_trait(trait, language):
                        if language == "Tiếng Việt":
                            return trait_translation.get(trait, trait)  # Trả về tiếng Việt hoặc giữ nguyên
                        return trait

                    # Generate nhận xét từ tất cả các traits
                    if language == "Tiếng Việt":
                        st.write("### Dựa vào các thông tin bạn cung cấp, để xem biểu đồ tài chính của bạn ra sao nhé")
                    else:
                        st.write("### Based on the information you provide, let’s see what your financial chart looks like!")
                    
                    # Sắp xếp các traits theo điểm từ cao đến thấp
                    sorted_traits = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
                    # Hàm mã hóa ảnh thành base64
                    def encode_image(image_path):
                        with open(image_path, "rb") as img_file:
                            return base64.b64encode(img_file.read()).decode()

                    # Đường dẫn đến ảnh tomi.png
                    # tomi_path = os.path.join("images", "Tomi.png")

                    # Mã hóa ảnh tomi.png
                    encoded_tomi = encode_image(tomi_path)
                    # Tạo biến thông báo spinner dựa trên ngôn ngữ
                    spinner_message = 'Đang đọc biểu đồ sao của bạn, chờ xíu nhe!' if language == "Tiếng Việt" else 'Creating your report, please wait...'
                    with st.spinner(spinner_message):
                        # Hiển thị ảnh trong khi chờ load
                        placeholder = st.empty()  # Tạo placeholder để quản lý nội dung
                        placeholder.markdown(
                            f"""
                            <div style="display: flex; justify-content: center; align-items: center; height: 300px;">
                                <img src="data:image/png;base64,{encoded_tomi}" width="200" class="floating">
                            </div>
                            <style>
                            .floating {{
                                animation: float 3s ease-in-out infinite;
                            }}
                            @keyframes float {{
                                0% {{ transform: translateY(0px); }}
                                50% {{ transform: translateY(-15px); }}
                                100% {{ transform: translateY(0px); }}
                            }}
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                        # Giả lập quá trình xử lý (ví dụ 3 giây)
                        time.sleep(3)

                    # Xóa hình ảnh sau khi quá trình load kết thúc
                    placeholder.empty()
                    # Nhận thông tin người dùng để tạo hash duy nhất
                    birth_date_str = birth_date.strftime("%Y-%m-%d")
                    birth_time_str = f"{hour:02}:{minute:02} {am_pm}"
                    birth_place_str = normalize_place(birth_place)  # Nơi sinh

                    # Tạo user_hash mới từ thông tin hiện tại của người dùng
                    current_user_hash = generate_user_hash(birth_date_str, birth_time_str, birth_place_str, language)
                    
                    # Khôi phục danh sách user_hash từ file (nếu chưa có)
                    if 'user_hash' not in st.session_state:
                        st.session_state.user_hash = load_user_hash()

                    # Kiểm tra nếu user_hash chưa có trong danh sách
                    if current_user_hash not in st.session_state.user_hash:
                        st.session_state.user_hash.append(current_user_hash)  # Thêm user_hash vào danh sách
                    #     save_user_hash(current_user_hash)  # Lưu user_hash mới vào file

                    # Kiểm tra nếu user_hash đã có trong cache
                    if current_user_hash in st.session_state.report_cache:
                        financial_traits_text, top_traits_description,eligible_content, necessary_content = st.session_state.report_cache[current_user_hash]

                        # Hiển thị lại nội dung từ cache
                        with st.expander("**Dựa vào Biểu Đồ, Tomi dự đoán rằng tính cách tài chính của bạn có thể có xu hướng sau:**" if language == "Tiếng Việt" else "###### **Based on the chart, Tomi can predict that your financial personality might tend to look like this:**", expanded=True):
                            # st.markdown(financial_traits_text, unsafe_allow_html=True)
                            st.markdown(
                                f"""
                                <div style="text-align: justify;">
                                    {markdown_to_html(financial_traits_text)}</div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.write("### Hmmm ... Vậy **tính cách** tài chính của tôi là gì nhỉ?" if language == "Tiếng Việt" else "### Hmmm ... So, what’s my financial tendencies?")
                        with st.expander("**Từ những thông tin trên, Tomi có thể thấy...**" if language == "Tiếng Việt" else "###### **From the above information, Tomi can tell that...**", expanded=True):
                            # st.write(top_traits_description)
                            st.markdown(
                                f"""
                                <div style="text-align: justify;">
                                    {markdown_to_html(top_traits_description)}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.write("                         ")
                        st.write("### Sản phẩm tài chính của Timo nào sẽ phù hợp với tôi đây?" if language == "Tiếng Việt" else "### Which Timo financial product would be the best fit for me?")
                        with st.expander("**Theo Tomi dự đoán...** " if language == "Tiếng Việt" else "**From what Tomi see...**", expanded=True):
                            # st.write(eligible_content)
                            st.markdown(
                                f"""
                                <div style="text-align: justify;">
                                    {markdown_to_html(eligible_content)}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            

                        # if language == "Tiếng Việt":
                        #     st.subheader("Bạn sẽ thích: ")
                        #     eligible_df['Score'] = eligible_df['Score'].round(2)
                        #     # st.markdown(eligible_df.to_html(classes='custom-table'), unsafe_allow_html=True)
                        #     eligible_df_display = eligible_df.drop(columns=['Score'])
                        #     st.markdown(eligible_df_display.to_html(classes='custom-table'), unsafe_allow_html=True)
                        # else:
                        #     st.subheader("Your matches:")
                        #     eligible_df['Score'] = eligible_df['Score'].round(2)  
                        #     # st.markdown(eligible_df.to_html(classes='custom-table'), unsafe_allow_html=True)
                        #     eligible_df_display = eligible_df.drop(columns=['Score'])
                        #     st.markdown(eligible_df_display.to_html(classes='custom-table'), unsafe_allow_html=True)
                        
                        st.write("                         ")
                        if necessary_content:
                            st.write("### Tôi nên sử dụng sản phẩm nào để tối ưu tài chính?" if language == "Tiếng Việt" else "### Which product should I use to optimize my finances?")
                            with st.expander("**Theo Tomi thì...**" if language == "Tiếng Việt" else "**From Tomi aspect...**", expanded=True):
                                # st.write(necessary_content)
                                st.markdown(
                                f"""
                                <div style="text-align: justify;">
                                    {markdown_to_html(necessary_content)}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            # if language == "Tiếng Việt":
                            #     st.subheader("Bạn sẽ cần: ")
                            #     necessary_df['Score'] = necessary_df['Score'].round(2) 
                            #     # st.markdown(necessary_df.to_html(classes='custom-table'), unsafe_allow_html=True)
                            #     necessary_df_display = necessary_df.drop(columns=['Score'])
                            #     st.markdown(necessary_df_display.to_html(classes='custom-table'), unsafe_allow_html=True)
                            # else:
                            #     st.subheader("You will need: ")
                            #     necessary_df['Score'] = necessary_df['Score'].round(2)  
                            #     # st.markdown(necessary_df.to_html(classes='custom-table'), unsafe_allow_html=True)
                            #     necessary_df_display = necessary_df.drop(columns=['Score'])
                            #     st.markdown(necessary_df_display.to_html(classes='custom-table'), unsafe_allow_html=True)
                                            # Sử dụng expander để ẩn/hiện phần nhận xét chi tiết
                        # Tải ảnh và mã hóa base64
                        # sticker_1 = base64.b64encode(open(sticker_1_path, "rb").read()).decode()
                        # sticker_2 = base64.b64encode(open(sticker_2_path, "rb").read()).decode()
                        # sticker_3 = base64.b64encode(open(sticker_3_path, "rb").read()).decode()
                        
                        st.write("                         ")
                        st.write("### Đánh Giá" if language == "Tiếng Việt" else "### Feedback")
                        # Tab Đánh giá
                        # with st.expander(feedback_message, expanded=True):
                        #     st.markdown(
                        #         f"""
                        #         <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">
                        #             <div style="flex: 1; display: flex; justify-content: center; flex-direction: column; align-items: center;">
                        #                 <a href="https://not-interested.streamlit.app/" target="_blank">
                        #                     <img src="data:image/png;base64,{sticker_1}" width="100" style="cursor: pointer;"/>
                        #                 </a>
                        #                 <p style="margin-top: 5px; font-size: 18px; text-align: center;">{texts[language][0]}</p>
                        #             </div>
                        #             <div style="flex: 1; display: flex; justify-content: center; flex-direction: column; align-items: center;">
                        #                 <a href="https://mehh-soso.streamlit.app/" target="_blank">
                        #                     <img src="data:image/png;base64,{sticker_2}" width="100" style="cursor: pointer;"/>
                        #                 </a>
                        #                 <p style="margin-top: 5px; font-size: 18px; text-align: center;">{texts[language][1]}</p>
                        #             </div>
                        #             <div style="flex: 1; display: flex; justify-content: center; flex-direction: column; align-items: center;">
                        #                 <a href="https://good-job.streamlit.app/" target="_blank">
                        #                     <img src="data:image/png;base64,{sticker_3}" width="100" style="cursor: pointer;"/>
                        #                 </a>
                        #                 <p style="margin-top: 5px; font-size: 18px; text-align: center;">{texts[language][2]}</p>
                        #             </div>
                        #         </div>
                        #         """,
                        #         unsafe_allow_html=True,
                        #     )
                    else:
                        # Tạo đoạn văn bản mô tả tất cả các traits với số thứ tự
                        financial_traits_text = ""
                        trait_colors = {
                            "Adventurous": "#2774ae",
                            "Convenient": "#9D03C7",
                            "Impulsive": "#d94b24",
                            "Conservative": "#ff9425",
                            "Cautious": "green",
                            "Analytical": "#f1d800",
                            "Mạo Hiểm": "#2774ae",  # Adventurous
                            "Tiện Lợi": "#9D03C7",  # Convenient
                            "Phóng Khoáng": "#d94b24",  # Impulsive
                            "Kiên Định": "#ff9425",  # Conservative
                            "Cẩn Trọng": "green",  # Cautious
                            "Tỉ Mỉ": "#f1d800"  # Analytical
                                                }
                        # '#2774ae', '#9D03C7','#d94b24', '#ff9425','green', '#f1d800'] 
                        start_time = time.time()
                        with st.spinner(spinner_message):
                            # Hiển thị ảnh trong khi chờ load
                            placeholder = st.empty()  # Tạo placeholder để quản lý nội dung
                            placeholder.markdown(
                                f"""
                                <div style="display: flex; justify-content: center; align-items: center; height: 300px;">
                                    <img src="data:image/png;base64,{encoded_tomi}" width="200" class="floating">
                                </div>
                                <style>
                                .floating {{
                                    animation: float 3s ease-in-out infinite;
                                }}
                                @keyframes float {{
                                    0% {{ transform: translateY(0px); }}
                                    50% {{ transform: translateY(-15px); }}
                                    100% {{ transform: translateY(0px); }}
                                }}
                                </style>
                                """,
                                unsafe_allow_html=True
                            )
                            # Giả lập quá trình xử lý báo cáo (ví dụ: chạy một số logic phức tạp)
                            time.sleep(10)  # Thay bằng logic tạo báo cáo thực tế

                            # Tính thời gian xử lý
                            end_time = time.time()
                            processing_time = end_time - start_time
                            print (processing_time)

                            # Xóa hình ảnh sau khi quá trình load kết thúc
                            placeholder.empty()
                            for index, (trait, score) in enumerate(sorted_traits):
                                if index == 0:
                                    note = " - cao nhất" if language == "Tiếng Việt" else " - highest"
                                elif index == len(sorted_traits) - 1:
                                    note = " - thấp nhất" if language == "Tiếng Việt" else " - lowest"
                                else:
                                    note = ""

                                # Lấy màu sắc dựa trên trait
                                trait = get_translated_trait(trait, language)
                                color = trait_colors.get(trait.title(), "black")  # Mặc định là "black" nếu không có trong trait_colors

                                description = get_trait_description_with_gpt(trait, score, language,age)
                                
                                # Sử dụng HTML để thay đổi màu sắc của trait
                                financial_traits_text += f"{index + 1}. <b style='color:{color};'>{trait.title()}</b> **({score:.3f}{note}):**\n\n {description}\n\n"

                            # Hiển thị toàn bộ đoạn văn bản
                            with st.expander("**Dựa vào Biểu Đồ, Tomi dự đoán rằng tính cách tài chính của bạn có thể có xu hướng sau:**" if language == "Tiếng Việt" else "###### **Based on the chart, Tomi can predict that your financial personality might tend to look like this:**", expanded=True):
                                # st.markdown(financial_traits_text, unsafe_allow_html=True)
                                st.markdown(
                                    f"""
                                    <div style="text-align: justify;">
                                        {markdown_to_html(financial_traits_text)}</div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            # Hiển thị mô tả cho top 3 traits
                            top_3_traits = get_top_3_traits(final_scores)
                            
                            top_traits_description = get_top_traits_description_with_gpt(top_3_traits,final_scores, language, age)

                            # product_content = get_product_recommendation_with_gpt(eligible_products, necessary_products, final_scores, language, age)
                            eligible_content = generate_recommendation_for_eligible(eligible_df, final_scores, language, age,product_hierarchy)
                            necessary_content = generate_recommendation_for_necessary(necessary_df, eligible_df, final_scores, language, age)

                            # Lưu báo cáo vào cache với user_hash
                            st.session_state.report_cache[current_user_hash] = (financial_traits_text, top_traits_description, eligible_content,necessary_content)
                            # Lưu báo cáo vào file txt để sử dụng lại khi khởi động lại ứng dụng
                            append_report_cache_to_txt(current_user_hash, financial_traits_text, top_traits_description,  eligible_content,necessary_content)
                        # Hiển thị tiêu đề nhận xét tổng quát dựa trên ngôn ngữ được chọn
                        st.write("### Hmmm ... Vậy **tính cách** tài chính của tôi là gì nhỉ?" if language == "Tiếng Việt" else "### Hmmm ... So, what’s my financial tendencies?")

                        # Sử dụng expander để ẩn/hiện phần nhận xét chi tiết
                        with st.expander("**Từ những thông tin trên, Tomi có thể thấy...**" if language == "Tiếng Việt" else "###### **From the above information, Tomi can tell that...**", expanded=True):                            # st.write(top_traits_description)
                            # st.write(top_traits_description)
                            st.markdown(
                                f"""
                                <div style="text-align: justify;">
                                    {markdown_to_html(top_traits_description)}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        
                        st.write("                         ")
                        # Hiển thị nội dung sản phẩm
                        st.write("### Sản phẩm tài chính của Timo nào sẽ phù hợp với tôi đây?" if language == "Tiếng Việt" else "### Which Timo financial product would be the best fit for me?")                        
                        with st.expander("**Theo Tomi dự đoán...** " if language == "Tiếng Việt" else "**From what Tomi see...**", expanded=True):
                            # st.write(eligible_content)
                            st.markdown(
                                f"""
                                <div style="text-align: justify;">
                                    {markdown_to_html(eligible_content)}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        # if language == "Tiếng Việt":
                        #     st.subheader("Bạn sẽ thích: ")
                        #     eligible_df['Score'] = eligible_df['Score'].round(2)
                        #     # st.markdown(eligible_df.to_html(classes='custom-table'), unsafe_allow_html=True)
                        #     eligible_df_display = eligible_df.drop(columns=['Score'])
                        #     st.markdown(eligible_df_display.to_html(classes='custom-table'), unsafe_allow_html=True)
                        # else:
                        #     st.subheader("Your matches:")
                        #     eligible_df['Score'] = eligible_df['Score'].round(2)  
                        #     # st.markdown(eligible_df.to_html(classes='custom-table'), unsafe_allow_html=True)
                        #     eligible_df_display = eligible_df.drop(columns=['Score'])
                        #     st.markdown(eligible_df_display.to_html(classes='custom-table'), unsafe_allow_html=True)
                        
                        st.write("                         ")

                         # Chỉ hiển thị nếu có sản phẩm cần thiết
                        if necessary_content:
                            st.write("### Tôi nên sử dụng sản phẩm nào để tối ưu tài chính?" if language == "Tiếng Việt" else "### Which product should I use to optimize my finances?")
                            with st.expander("**Theo Tomi thì...**" if language == "Tiếng Việt" else "**From Tomi aspect...**", expanded=True):
                                # st.write(necessary_content)
                                st.markdown(
                                f"""
                                <div style="text-align: justify;">
                                    {markdown_to_html(necessary_content)}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                            # if language == "Tiếng Việt":
                            #     st.subheader("Bạn sẽ cần: ")
                            #     necessary_df['Score'] = necessary_df['Score'].round(2) 
                            #     # st.markdown(necessary_df.to_html(classes='custom-table'), unsafe_allow_html=True)
                            #     necessary_df_display = necessary_df.drop(columns=['Score'])
                            #     st.markdown(necessary_df_display.to_html(classes='custom-table'), unsafe_allow_html=True)
                            # else:
                            #     st.subheader("You will need: ")
                            #     necessary_df['Score'] = necessary_df['Score'].round(2)  
                            #     # st.markdown(necessary_df.to_html(classes='custom-table'), unsafe_allow_html=True)
                            #     necessary_df_display = necessary_df.drop(columns=['Score'])
                            #     st.markdown(necessary_df_display.to_html(classes='custom-table'), unsafe_allow_html=True)
                    
                        # Nếu không có trong cache, tạo nội dung sản phẩm mới với GPT
                            eligible_products = eligible_df[['Product', 'Label']].values.tolist()
                            necessary_products = necessary_df[['Product', 'Score', 'Label']].values.tolist()
                        
                        # Sử dụng expander để ẩn/hiện phần nhận xét chi tiết
                        # Tải ảnh và mã hóa base64
                        # sticker_1 = base64.b64encode(open(sticker_1_path, "rb").read()).decode()
                        # sticker_2 = base64.b64encode(open(sticker_2_path, "rb").read()).decode()
                        # sticker_3 = base64.b64encode(open(sticker_3_path, "rb").read()).decode()
                        
                        # st.write("                         ")
                        # st.write("### Đánh Giá" if language == "Tiếng Việt" else "### Feedback")
                        
                        # Tab Đánh giá
                        # with st.expander(feedback_message, expanded=True):
                        #     st.markdown(
                        #         f"""
                        #         <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">
                        #             <div style="flex: 1; display: flex; justify-content: center; flex-direction: column; align-items: center;">
                        #                 <a href="https://not-interested.streamlit.app/" target="_blank">
                        #                     <img src="data:image/png;base64,{sticker_1}" width="100" style="cursor: pointer;"/>
                        #                 </a>
                        #                 <p style="margin-top: 5px; font-size: 18px; text-align: center;">{texts[language][0]}</p>
                        #             </div>
                        #             <div style="flex: 1; display: flex; justify-content: center; flex-direction: column; align-items: center;">
                        #                 <a href="https://mehh-soso.streamlit.app/" target="_blank">
                        #                     <img src="data:image/png;base64,{sticker_2}" width="100" style="cursor: pointer;"/>
                        #                 </a>
                        #                 <p style="margin-top: 5px; font-size: 18px; text-align: center;">{texts[language][1]}</p>
                        #             </div>
                        #             <div style="flex: 1; display: flex; justify-content: center; flex-direction: column; align-items: center;">
                        #                 <a href="https://good-job.streamlit.app/" target="_blank">
                        #                     <img src="data:image/png;base64,{sticker_3}" width="100" style="cursor: pointer;"/>
                        #                 </a>
                        #                 <p style="margin-top: 5px; font-size: 18px; text-align: center;">{texts[language][2]}</p>
                        #             </div>
                        #         </div>
                        #         """,
                        #         unsafe_allow_html=True,
                        #     )
                        
            # Tab Feedback với các sticker
            with tabs[1]:  # Tab Đánh giá/Feedback
                st.header(tab_titles[1])  # Hiển thị tiêu đề Tab
                # st.write(feedback_message)  # Hiển thị link nhận xét


                # Tải ảnh và mã hóa base64
                # sticker_1 = base64.b64encode(open(sticker_1_path, "rb").read()).decode()
                # sticker_2 = base64.b64encode(open(sticker_2_path, "rb").read()).decode()
                # sticker_3 = base64.b64encode(open(sticker_3_path, "rb").read()).decode()

                # Tab Đánh giá
            #     with st.expander(feedback_message, expanded=True):
            #         st.markdown(
            #             f"""
            #             <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">
            #                 <div style="flex: 1; display: flex; justify-content: center; flex-direction: column; align-items: center;">
            #                     <a href="https://not-interested.streamlit.app/" target="_blank">
            #                         <img src="data:image/png;base64,{sticker_1}" width="100" style="cursor: pointer;"/>
            #                     </a>
            #                     <p style="margin-top: 5px; font-size: 18px; text-align: center;">{texts[language][0]}</p>
            #                 </div>
            #                 <div style="flex: 1; display: flex; justify-content: center; flex-direction: column; align-items: center;">
            #                     <a href="https://mehh-soso.streamlit.app/" target="_blank">
            #                         <img src="data:image/png;base64,{sticker_2}" width="100" style="cursor: pointer;"/>
            #                     </a>
            #                     <p style="margin-top: 5px; font-size: 18px; text-align: center;">{texts[language][1]}</p>
            #                 </div>
            #                 <div style="flex: 1; display: flex; justify-content: center; flex-direction: column; align-items: center;">
            #                     <a href="https://good-job.streamlit.app/" target="_blank">
            #                         <img src="data:image/png;base64,{sticker_3}" width="100" style="cursor: pointer;"/>
            #                     </a>
            #                     <p style="margin-top: 5px; font-size: 18px; text-align: center;">{texts[language][2]}</p>
            #                 </div>
            #             </div>
            #             """,
            #             unsafe_allow_html=True,
            #         )

            # st.write("                         ")

    # Nút Refresh để làm mới ứng dụng
if st.button(f"{refresh_button_label}"):
         st.experimental_set_query_params(reload="true")



# Hàm mã hóa ảnh thành chuỗi base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Đường dẫn tới ảnh QR code và các nút store
# qr_path = os.path.join("images", "qr.png")
# app_store_path = os.path.join("images", "ast.png")
# google_play_path =os.path.join("images", "adr.png")
# Mã hóa các ảnh thành base64
# encoded_qr = encode_image(qr_path)
# encoded_app_store = encode_image(app_store_path)
# encoded_google_play = encode_image(google_play_path)

# CSS tùy chỉnh cho section cuối cùng của trang
st.markdown(
    """
    <style>
    .end-section {
        # background-image: url('https://images.newscientist.com/wp-content/uploads/2023/07/03093742/sei162306009.jpg');
        background-position: center;
        # background-size: cover;
        padding: 30px;
        margin-top: 100px;  /* Tạo khoảng cách với nội dung phía trên */
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        flex-wrap: wrap;  /* Đảm bảo nội dung không bị tràn */
    }
    .content-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
    }
    .content-center {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 10px;
    }
    .footer-text {
        width: 100%;
        text-align: center;
        color: white;
        font-size: 18px;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Nội dung cho từng ngôn ngữ
footer_text = (
    "Tải ứng dụng ngay để trải nghiệm sản phẩm được gợi ý bởi Astrotomi!"
    if language == "Tiếng Việt"
    else "Download the app now to try our products recommended by Astrotomi!"
)

# HTML hiển thị QR code và các nút tải app theo bố cục
# st.markdown(
#     f"""
#     <div class="end-section">
#         <div>End section here</div>
#         <!--<div class="content-row">
#             <div class="content-center">
#                 <img src="data:image/png;base64,{encoded_qr}" width="150" style="cursor: pointer;" onclick="openModal()"/>
#             </div>
#             <div class="content-center">
#                 <a href="https://app.adjust.com/1h1f5pz7" target="_blank">
#                     <img src="data:image/png;base64,{encoded_app_store}" width="160" style="cursor: pointer;"/>
#                 </a>
#                 <a href="https://app.adjust.com/1hccdzw5?fallback=https%3A%2F%2Fplay.google.com%2Fstore%2Fapps%2Fdetails%3Fid%3Dio.lifestyle.plus%26hl%3D&redirect_macos=https%3A%2F%2Fplay.google.com%2Fstore%2Fapps%2Fdetails%3Fid%3Dio.lifestyle.plus%26hl%3D" target="_blank">
#                     <img src="data:image/png;base64,{encoded_google_play}" width="160" style="cursor: pointer;"/>
#                 </a>
#             </div>
#         </div>
#         <div class="footer-text">{footer_text}</div>-->
#     </div>
#     """,
#     unsafe_allow_html=True
# )
# Nội dung footer theo ngôn ngữ
if language == "Tiếng Việt":
    footer_content = """Được phát triển bởi ...."""
else:
    footer_content = """🌟 Developed by the Timo team | <a style='color: #FFD700;' href="https://timo.vn" target="_blank">Timo.vn</a> 🌟"""

# CSS và HTML cho footer
footer = f"""
    <style>
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #6A0DAD;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 16px;
        z-index: 100;
    }}
    </style>
    <!--<div class="footer">
        <p>{footer_content}</p>
    </div>-->
"""

# Hiển thị footer trên Streamlit
st.markdown(footer, unsafe_allow_html=True)

# # Hàm xóa cache từ session state và file txt
# def delete_cache_by_user_hash():
#     if 'report_cache' not in st.session_state:
#         st.session_state['report_cache'] = {}
    
#     # Nhập user_hash để xóa cache
#     user_hash_input = st.text_input("Enter User Hash to delete cache:")
    
#     if st.button("Delete Cache"):
#         # Xóa cache trong session state
#         if user_hash_input in st.session_state['report_cache']:
#             del st.session_state['report_cache'][user_hash_input]
#             st.success(f"Cache for user_hash: {user_hash_input} has been deleted from session.")
#         else:
#             st.warning(f"No cache found for user_hash: {user_hash_input} in session.")
        
#         # Xóa cache từ file txt
#         if delete_cache_from_txt(user_hash_input):
#             st.success(f"Cache for user_hash: {user_hash_input} has been deleted from file.")
#         else:
#             st.warning(f"No cache found for user_hash: {user_hash_input} in file.")

# # Hàm xóa cache khỏi file report_cache.txt
# def delete_cache_from_txt(user_hash_to_delete):
#     try:
#         # Đọc tất cả các dòng từ file
#         with open('txt3.txt', 'r') as file:
#             lines = file.readlines()
        
#         # Viết lại file, bỏ qua user_hash cần xóa
#         with open('txt3.txt', 'w') as file:
#             cache_found = False
#             for line in lines:
#                 if not line.startswith(user_hash_to_delete):  # Nếu không phải là user_hash cần xóa
#                     file.write(line)
#                 else:
#                     cache_found = True
        
#         return cache_found  # Trả về True nếu tìm thấy và xóa user_hash, False nếu không
#     except FileNotFoundError:
#         return False

# # *** Thêm phần Admin Access và Cache Management ***
# ADMIN_PASSWORD = "admin123"  

import streamlit as st

# Nội dung Tiếng Việt
about_us_vn = """
Đôi lời gì đó chưa biết viết gì.
"""

# Nội dung Tiếng Anh
about_us_en = """
We are a team of developers who are passionate about creating useful and innovative applications.
"""

# Hàm mã hóa ảnh thành base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Mã hóa ảnh QR và Tomi
# encoded_qr = encode_image("images/qr.png")
# encoded_tomi = encode_image("images/Tomi.png")

# CSS cho căn giữa và tạo animation cho Tomi
st.markdown(
    """
    <style>
    .about-us {
        text-align: justify;
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 20px;
    }
    .about-title {
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 10px;
        text-align: center;
    }
    .about-content {
        width: 80%;  /* Điều chỉnh độ rộng */
        text-align: justify;
        line-height: 1.6;
    }
    .sidebar-images {
        margin-top: 20px;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 15px;
    }
    .tomi-img {
        animation: floating 3s infinite ease-in-out;
    }
    @keyframes floating {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Nội dung "About Us" tùy theo ngôn ngữ
if language == "Tiếng Việt":
    title = "VỀ CHÚNG TÔI"
    content = about_us_vn
else:
    title = "ABOUT US"
    content = about_us_en

# Hiển thị tiêu đề, nội dung, và ảnh Tomi + QR code
# st.sidebar.markdown(
#     f"""
#     <div class="about-us">
#         <div class="about-title">{title}</div>
#         <div class="about-content">{content}</div>
#         <!--<div class="sidebar-images">
#             <img class="tomi-img" src="data:image/png;base64,{encoded_tomi}" width="120"/>
#             <img src="data:image/png;base64,{encoded_qr}" width="120"/>
#         </div>-->
#     </div>
#     """,
#     unsafe_allow_html=True
# )




# # Hàm kiểm tra đăng nhập admin
# def admin_access():
#     st.sidebar.subheader("Admin Access")
#     admin_password_input = st.sidebar.text_input("Enter Admin Password:", type="password")
    
#     if st.sidebar.button("Login as Admin"):
#         if admin_password_input == ADMIN_PASSWORD:
#             st.session_state['is_admin'] = True
#             st.sidebar.success("Logged in as Admin")
#         else:
#             st.sidebar.error("Invalid Admin Password")

# # Hàm đăng xuất admin
# def admin_logout():
#     if st.sidebar.button("Logout"):
#         st.session_state['is_admin'] = False
#         st.sidebar.success("Logged out successfully.")

# # Kiểm tra xem admin đã đăng nhập chưa
# if 'is_admin' not in st.session_state:
#     st.session_state['is_admin'] = False

# # Giao diện cho admin sau khi đăng nhập
# if st.session_state['is_admin']:
#     # Thêm các tab cho admin quản lý
#     tab1, tab2 = st.tabs(["User Dashboard", "Admin Panel"])

#     # Tab hiển thị danh sách user_hash
#     with tab1:
#         st.subheader("Saved User Hashes")
#         if 'report_cache' in st.session_state:
#             if len(st.session_state['report_cache']) > 0:
#                 for user_hash in st.session_state['report_cache'].keys():
#                     st.write(user_hash)
#             else:
#                 st.write("No user hash found.")
#         else:
#             st.write("No user hash found.")
    
#     # Tab admin panel để xóa cache và logout
#     with tab2:
#         st.subheader("Admin Panel")
#         delete_cache_by_user_hash()  # Xóa cache
#         admin_logout()  # Đăng xuất admin
# else:
#     # Nếu chưa đăng nhập, hiển thị giao diện đăng nhập
#     admin_access()
