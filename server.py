import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommendation System")

DATA_URL = "books.csv"

@st.cache_data
def load_data(nrows=15000):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    def lowercase(x): return str(x).lower()
    data.columns = data.columns.str.strip()
    data = data.rename(lowercase, axis='columns', inplace=False)
    data = data.dropna(axis='index', inplace=False)
    data = data.reset_index(drop=False)
    print(data.info())
    return data


def combined_features(row):
    return str(row['rating'])+" "+str(row['genre'])+" "+str(row['author'])+" "+str(row['totalratings'])


def Cosine_Similarity(count_matrix):
    return cosine_similarity(count_matrix)


def get_index_from_title(data, title):
    '''Getting index of a specific title''' 
    try:
        index = data[data.title == title]["index"].values[0]
        return index
    except:
        st.write(f"<hr />", unsafe_allow_html=True)
        st.write(f"<h4>There are no similar books</h4>", unsafe_allow_html=True)


def get_title_from_index(data, index):
    '''Title from an index'''
    return data[data.index == index]["title"].values[0]


def get_all_details_from_index(data, index):
    '''Getting various details from an index'''
    title = data[data.index == index]["title"].values[0]
    description = data[data.index == index]["desc"].values[0]
    rating = data[data.index == index]["rating"].values[0]
    img = data[data.index == index]["img"].values[0]
    link = data[data.index == index]["link"].values[0]
    genres = data[data.index == index]["genre"].values[0]
    return [title, description, rating, img, link, genres]


def detailed_view(title, description, rating, image, link, genres):
    '''Detailed Description'''
    st.write(f"<hr />", unsafe_allow_html=True)
    st.write(f"<h4><em>{title}</em></h4>", unsafe_allow_html=True)
    st.write(f"<br />", unsafe_allow_html=True)
    st.image(image, width=200)
    st.write(f"<br />", unsafe_allow_html=True)
    st.write(f"<p>{description}</p>", unsafe_allow_html=True)
    st.write(f"<p><b>Genres: </b>{genres}</p>", unsafe_allow_html=True)
    st.write(f"<p><b>Rating: </b>{rating}/5</p>", unsafe_allow_html=True)
    st.write(f'<a href="{link}" target="_blank">Learn more</a>', unsafe_allow_html=True)
    return ""


def recommend_books(cosine_sim, data):
    '''Recommending books'''
    book_user_likes = "Between Two Fires: American Indians in the Civil War"
    print(data.columns)

    # User Input
    user_input = st.text_input("Enter the book title that you like", book_user_likes)
    print(user_input)

    if (user_input != None):
        book_index = get_index_from_title(data, user_input)
        similar_books = list(enumerate(cosine_sim[book_index]))

        sorted_similar_books = sorted(
            similar_books, key=lambda x: x[1], reverse=True)

        if (len(similar_books) > 1):
            i = 0
            for book in sorted_similar_books:
                [title, description, rating, image, link, genres] = get_all_details_from_index(data, int(book[0]))
                st.write(detailed_view(
                   title,
                   description,
                   rating,
                   image,
                   link,
                   genres
                )) 
                i = i+1
                if i > 15:
                    break


def main():
    '''Main function'''
    # Hide the Streamlit menu and footer
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("Book Recommendation System")
    st.write(f"<br />", unsafe_allow_html=True)
    data = load_data(20000)
    data["combined_features"] = data.apply(combined_features, axis=1)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data["combined_features"])
    cosine_sim = Cosine_Similarity(count_matrix)
    print("Count Matrix:", count_matrix.toarray())
    print(cosine_sim)
    recommend_books(cosine_sim=cosine_sim, data=data)


if __name__ == "__main__":
    main()