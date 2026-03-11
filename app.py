import streamlit as st

from recommender import MovieRecommender


st.set_page_config(page_title="Movie Recommender", page_icon=":clapper:", layout="wide")
st.title("Movie Recommender")
st.caption("Content-based movie recommendations")


@st.cache_resource
def load_model():
    return MovieRecommender("data_ass2_part2_wk5.csv")

model = load_model()

with st.form("recommendation_form", clear_on_submit=False):
    left, right = st.columns([3, 1])
    with left:
        user_title = st.text_input("Enter a movie title", value="")
    with right:
        top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    if not user_title.strip():
        st.warning("Please enter a movie title.")
    else:
        matched, result = model.get_recommendations(user_title, n=top_n)
        if result is None:
            st.error("Title not found. Try another title or a partial match.")
        else:
            if matched != user_title.strip().lower():
                st.info(f"Matched to: '{matched}'")

            st.subheader(f"Top {top_n} recommendations for '{user_title}'")
            display_score_col = "chances you'll like it (based on similiarity score)"
            if "similarity_score (%)" in result.columns:
                result = result.rename(columns={"similarity_score (%)": display_score_col})

            score_col = None
            for candidate in [display_score_col, "similarity_score (%)"]:
                if candidate in result.columns:
                    score_col = candidate
                    break

            if score_col in result.columns:
                result = result.sort_values(by=score_col, ascending=False).reset_index(drop=True)

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True,
                column_config={
                    score_col: st.column_config.NumberColumn(score_col, format="%.2f%%")
                } if score_col in result.columns else None,
            )