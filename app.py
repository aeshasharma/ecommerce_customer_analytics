import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from pathlib import Path


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="E-Commerce Customer Analytics",
    page_icon="🛒",
    layout="wide"
)


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(
    __file__
).resolve().parent


DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR = BASE_DIR / "outputs"


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
    }

    .sub-title {
        color: #777;
        font-size: 18px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 15px;
        background: #f8f9fa;
        border: 1px solid #eeeeee;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_DIR / "Ecommerce.csv"
    )


df = load_data()


# ==================================================
# LOAD SEGMENTS
# ==================================================

segment_file = (
    OUTPUT_DIR /
    "customer_segments.csv"
)


if segment_file.exists():

    segments = pd.read_csv(
        segment_file
    )

else:

    segments = None


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">'
    '🛒 E-Commerce Customer Analytics'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="sub-title">'
    'Analyze customer behavior, predict spending, '
    'purchase probability and customer segments.'
    '</div>',
    unsafe_allow_html=True
)


st.divider()


# ==================================================
# TOP METRICS
# ==================================================

total_customers = (
    df["customer_id"]
    .nunique()
)


total_revenue = (
    df["revenue"]
    .sum()
)


total_purchases = (
    df["purchased"]
    .sum()
)


purchase_rate = (
    df["purchased"]
    .mean() * 100
)


col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "👥 Customers",
        f"{total_customers:,}"
    )


with col2:

    st.metric(
        "💰 Revenue",
        f"₹{total_revenue:,.2f}"
    )


with col3:

    st.metric(
        "🛍️ Purchases",
        f"{total_purchases:,}"
    )


with col4:

    st.metric(
        "📈 Purchase Rate",
        f"{purchase_rate:.2f}%"
    )


st.divider()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title(
    "🔮 Customer Prediction"
)


unit_price = st.sidebar.number_input(
    "Unit Price",
    min_value=0.0,
    value=1000.0,
    step=50.0
)


quantity = st.sidebar.number_input(
    "Quantity",
    min_value=1,
    value=1,
    step=1
)


discount = st.sidebar.slider(
    "Discount %",
    min_value=0,
    max_value=100,
    value=10
)


pages_viewed = st.sidebar.number_input(
    "Pages Viewed",
    min_value=0,
    value=10,
    step=1
)


time_on_site = st.sidebar.number_input(
    "Time on Site (seconds)",
    min_value=0,
    value=600,
    step=10
)


added_to_cart = st.sidebar.selectbox(
    "Added to Cart",
    [0, 1]
)


cart_abandoned = st.sidebar.selectbox(
    "Cart Abandoned",
    [0, 1]
)


rating = st.sidebar.slider(
    "Rating",
    min_value=1,
    max_value=5,
    value=4
)


# ==================================================
# PREDICTION
# ==================================================

if st.sidebar.button(
    "🔮 Predict"
):

    # ----------------------------------------------
    # LINEAR REGRESSION
    # ----------------------------------------------

    linear_model = joblib.load(
        MODEL_DIR /
        "linear_regression.pkl"
    )


    linear_input = pd.DataFrame({

        "unit_price": [
            unit_price
        ],

        "quantity": [
            quantity
        ],

        "discount_percent": [
            discount
        ],

        "pages_viewed": [
            pages_viewed
        ],

        "time_on_site_sec": [
            time_on_site
        ],

        "added_to_cart": [
            added_to_cart
        ]

    })


    predicted_revenue = (
        linear_model
        .predict(
            linear_input
        )[0]
    )


    # ----------------------------------------------
    # LOGISTIC REGRESSION
    # ----------------------------------------------

    logistic_model = joblib.load(
        MODEL_DIR /
        "logistic_regression.pkl"
    )


    logistic_input = pd.DataFrame({

        "unit_price": [
            unit_price
        ],

        "quantity": [
            quantity
        ],

        "discount_percent": [
            discount
        ],

        "pages_viewed": [
            pages_viewed
        ],

        "time_on_site_sec": [
            time_on_site
        ],

        "added_to_cart": [
            added_to_cart
        ],

        "cart_abandoned": [
            cart_abandoned
        ],

        "rating": [
            rating
        ]

    })


    purchase_probability = (
        logistic_model
        .predict_proba(
            logistic_input
        )[0][1]
    )


    # ----------------------------------------------
    # RESULTS
    # ----------------------------------------------

    st.header(
        "🔮 Prediction Results"
    )


    result1, result2 = (
        st.columns(2)
    )


    with result1:

        st.metric(
            "Predicted Customer Spending",
            f"₹{max(0, predicted_revenue):,.2f}"
        )


    with result2:

        st.metric(
            "Purchase Probability",
            f"{purchase_probability * 100:.2f}%"
        )


    if purchase_probability >= 0.5:

        st.success(
            "✅ Customer is likely to purchase."
        )

    else:

        st.warning(
            "⚠️ Customer is unlikely to purchase."
        )


    st.divider()


# ==================================================
# ANALYTICS TABS
# ==================================================

st.header(
    "📊 Analytics Dashboard"
)


tab1, tab2, tab3, tab4 = st.tabs([

    "💰 Revenue",

    "🛍️ Products",

    "👥 Customers",

    "🎯 Customer Segments"

])


# ==================================================
# TAB 1 — REVENUE
# ==================================================

with tab1:

    st.subheader(
        "Revenue Analysis"
    )


    revenue_category = (
        df.groupby(
            "product_category"
        )["revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
    )


    fig = px.bar(

        x=revenue_category.index,

        y=revenue_category.values,

        labels={
            "x": "Product Category",
            "y": "Revenue"
        },

        title="Revenue by Product Category"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==================================================
# TAB 2 — PRODUCTS
# ==================================================

with tab2:

    st.subheader(
        "Product Analysis"
    )


    product_counts = (
        df[
            "product_category"
        ]
        .value_counts()
    )


    fig = px.bar(

        x=product_counts.index,

        y=product_counts.values,

        labels={
            "x": "Product Category",
            "y": "Number of Sessions"
        },

        title="Popular Product Categories"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader(
        "Top Products by Revenue"
    )


    top_products = (
        df.groupby(
            "product_id"
        )["revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )


    st.dataframe(
        top_products
        .reset_index()
        .rename(
            columns={
                "product_id":
                    "Product ID",

                "revenue":
                    "Revenue"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# TAB 3 — CUSTOMERS
# ==================================================

with tab3:

    st.subheader(
        "Customer Purchase Behavior"
    )


    purchase_summary = (
        df.groupby(
            "purchased"
        )["revenue"]
        .sum()
    )


    purchase_summary.index = [
        "Not Purchased"
        if x == 0
        else "Purchased"
        for x in purchase_summary.index
    ]


    fig = px.pie(

        values=purchase_summary.values,

        names=purchase_summary.index,

        title="Purchase vs Non-Purchase Revenue"

    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader(
        "Browsing Behavior"
    )


    fig2 = px.scatter(

        df.sample(
            min(3000, len(df)),
            random_state=42
        ),

        x="pages_viewed",

        y="time_on_site_sec",

        color="purchased",

        size="quantity",

        hover_data=[
            "customer_id",
            "product_id",
            "revenue"
        ],

        title="Browsing Behavior vs Purchase"

    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ==================================================
# TAB 4 — CUSTOMER SEGMENTS
# ==================================================

with tab4:

    st.header(
        "Customer Segments"
    )


    st.write(
        "RFM + K-Means based customer segmentation."
    )


    if segments is None:

        st.error(
            "customer_segments.csv not found."
        )

        st.info(
            "Run: "
            "python -m models.kmeans_clustering"
        )

    else:

        # ------------------------------------------
        # SEGMENT FILTER
        # ------------------------------------------

        available_segments = sorted(
            segments[
                "segment"
            ].dropna().unique()
        )


        selected_segments = st.multiselect(

            "Select Customer Segments",

            options=available_segments,

            default=available_segments

        )


        filtered_segments = segments[
            segments[
                "segment"
            ].isin(
                selected_segments
            )
        ]


        # ------------------------------------------
        # SEGMENT COUNTS
        # ------------------------------------------

        st.subheader(
            "Segment Overview"
        )


        segment_counts = (
            segments[
                "segment"
            ]
            .value_counts()
            .to_dict()
        )


        c1, c2, c3, c4, c5 = (
            st.columns(5)
        )


        with c1:

            st.metric(
                "🏆 VIP",
                segment_counts.get(
                    "VIP",
                    0
                )
            )


        with c2:

            st.metric(
                "🔵 Regular",
                segment_counts.get(
                    "Regular",
                    0
                )
            )


        with c3:

            st.metric(
                "🆕 New Customer",
                segment_counts.get(
                    "New Customer",
                    0
                )
            )


        with c4:

            st.metric(
                "⚠️ At Risk",
                segment_counts.get(
                    "At Risk",
                    0
                )
            )


        with c5:

            st.metric(
                "🔴 Lost",
                segment_counts.get(
                    "Lost",
                    0
                )
            )


        st.divider()


        # ------------------------------------------
        # BUBBLE CHART
        # ------------------------------------------

        st.subheader(
            "Customer Segmentation Analysis"
        )


        fig = px.scatter(

            filtered_segments,

            x="frequency",

            y="monetary",

            size="activity_score",

            color="segment",

            hover_name="customer_id",

            hover_data={

                "recency": True,

                "frequency": True,

                "monetary": ":.2f",

                "activity_score": ":.2f",

                "segment": True

            },

            labels={

                "frequency":
                    "Frequency",

                "monetary":
                    "Monetary",

                "recency":
                    "Recency",

                "segment":
                    "Segment",

                "activity_score":
                    "Activity Score"

            },

            color_discrete_map={

                "VIP":
                    "#5B9BD5",

                "Regular":
                    "#0066B3",

                "New Customer":
                    "#D89A9A",

                "Lost":
                    "#C62828",

                "At Risk":
                    "#59A96A"

            },

            title="Customer Segments"

        )


        fig.update_traces(
            marker={
                "opacity": 0.65,
                "line": {
                    "width": 1
                }
            }
        )


        fig.update_layout(

            height=600,

            template="plotly_dark",

            xaxis_title="Frequency",

            yaxis_title="Monetary",

            legend_title="Segment",

            margin=dict(
                l=40,
                r=40,
                t=70,
                b=40
            )

        )


        st.plotly_chart(

            fig,

            use_container_width=True

        )


        # ------------------------------------------
        # SEGMENT SUMMARY
        # ------------------------------------------

        st.subheader(
            "Segment Details"
        )


        summary = (
            segments
            .groupby("segment")
            .agg(

                Customers=(
                    "customer_id",
                    "count"
                ),

                Avg_Recency=(
                    "recency",
                    "mean"
                ),

                Avg_Frequency=(
                    "frequency",
                    "mean"
                ),

                Avg_Monetary=(
                    "monetary",
                    "mean"
                )

            )
            .reset_index()
        )


        summary = summary.rename(
            columns={
                "segment":
                    "Segment",

                "Avg_Recency":
                    "Average Recency",

                "Avg_Frequency":
                    "Average Frequency",

                "Avg_Monetary":
                    "Average Monetary"
            }
        )


        summary[
            "Average Recency"
        ] = summary[
            "Average Recency"
        ].round(1)


        summary[
            "Average Frequency"
        ] = summary[
            "Average Frequency"
        ].round(1)


        summary[
            "Average Monetary"
        ] = summary[
            "Average Monetary"
        ].round(2)


        st.dataframe(

            summary,

            use_container_width=True,

            hide_index=True

        )


        # ------------------------------------------
        # CUSTOMER TABLE
        # ------------------------------------------

        st.subheader(
            "Customer Segment Data"
        )


        st.dataframe(

            filtered_segments[

                [
                    "customer_id",

                    "recency",

                    "frequency",

                    "monetary",

                    "segment"

                ]

            ],

            use_container_width=True,

            hide_index=True

        )


# ==================================================
# RAW DATA
# ==================================================

st.divider()


with st.expander(
    "📄 View Original Dataset"
):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )