import streamlit as st
import numpy as np
import re
import textwrap
from pathlib import Path

from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import tensorflow as tf


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="S.L.E.D. | AI Lupus Screening",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE NAMES
# ============================================================

APP_DIR = Path(__file__).resolve().parent
CV_MODEL_PATH = APP_DIR / "keras_model.h5"
CV_LABELS_PATH = APP_DIR / "labels.txt"


def render_html(markup):
    st.html(
        textwrap.dedent(markup).strip()
    )


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
    <style>

    /* =========================
       GENERAL PAGE
       ========================= */

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(190, 170, 255, 0.20), transparent 25%),
            radial-gradient(circle at 90% 15%, rgba(120, 220, 255, 0.18), transparent 25%),
            radial-gradient(circle at 50% 90%, rgba(255, 180, 220, 0.15), transparent 30%),
            #f8f7ff;
    }

    html, body, [class*="css"] {
        font-family: "Arial", sans-serif;
        font-size: 19px;
    }

    /* Main text */

    p, li, label {
        font-size: 19px !important;
        line-height: 1.6 !important;
    }

    /* =========================
       BIG HEADINGS
       ========================= */

    h1 {
        font-size: 52px !important;
        font-weight: 900 !important;
        letter-spacing: -1px;
    }

    h2 {
        font-size: 35px !important;
        font-weight: 850 !important;
        margin-top: 25px !important;
    }

    h3 {
        font-size: 27px !important;
        font-weight: 800 !important;
    }

    /* =========================
       HERO
       ========================= */

    .hero {
        padding: 40px;
        border-radius: 30px;
        background:
            linear-gradient(
                135deg,
                rgba(120, 88, 255, 0.95),
                rgba(70, 180, 255, 0.92)
            );
        color: white;
        text-align: center;
        box-shadow: 0 15px 45px rgba(70, 80, 180, 0.25);
        margin-bottom: 30px;
        animation: floatCard 4s ease-in-out infinite;
    }

    .hero-title {
        font-size: 58px;
        font-weight: 950;
        letter-spacing: -2px;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 23px;
        font-weight: 600;
        opacity: 0.95;
    }

    .butterfly {
        font-size: 58px;
        margin-bottom: 5px;
        animation: butterflyFloat 3s ease-in-out infinite;
    }

    /* =========================
       SECTION CARDS
       ========================= */

    .section-card {
        background: rgba(255,255,255,0.88);
        border-radius: 25px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(120,100,220,0.15);
        box-shadow: 0 10px 30px rgba(80,70,130,0.10);
        transition: 0.3s ease;
    }

    .section-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 40px rgba(80,70,130,0.17);
    }

    /* =========================
       FEATURE CARDS
       ========================= */

    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 22px;
        min-height: 180px;
        box-shadow: 0 8px 25px rgba(70,70,120,0.10);
        border: 1px solid rgba(100,100,200,0.10);
        transition: all 0.25s ease;
    }

    .feature-card:hover {
        transform: scale(1.025);
    }

    .feature-icon {
        font-size: 40px;
    }

    .feature-title {
        font-size: 24px;
        font-weight: 850;
        margin: 8px 0;
    }

    .feature-text {
        font-size: 17px;
        color: #555;
    }

    /* =========================
       RESULT CARDS
       ========================= */

    .result-card {
        padding: 30px;
        border-radius: 28px;
        background: white;
        box-shadow: 0 12px 35px rgba(60,60,120,0.14);
        text-align: center;
        margin-top: 20px;
    }

    .result-number {
        font-size: 60px;
        font-weight: 950;
        margin: 5px 0;
    }

    .result-label {
        font-size: 21px;
        font-weight: 750;
    }

    /* =========================
       DOCTOR CARD
       ========================= */

    .doctor-card {
        margin-top: 35px;
        padding: 30px;
        border-radius: 25px;
        background: linear-gradient(
            135deg,
            #fff8fc,
            #f0f5ff
        );
        border: 2px solid rgba(100,100,220,0.15);
        box-shadow: 0 10px 30px rgba(80,70,130,0.10);
        text-align: center;
    }

    .doctor-title {
        font-size: 30px;
        font-weight: 900;
    }

    .doctor-text {
        font-size: 19px;
        line-height: 1.7;
    }

    /* =========================
       INFO BADGE
       ========================= */

    .badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 50px;
        background: #eeeaff;
        color: #5140a0;
        font-size: 16px;
        font-weight: 800;
        margin: 5px;
    }

    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        width: 100%;
        border-radius: 18px;
        min-height: 55px;
        font-size: 20px !important;
        font-weight: 850 !important;
        border: none;
        transition: 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 10px 25px rgba(70,70,150,0.20);
    }

    /* =========================
       INPUTS
       ========================= */

    div[data-baseweb="select"] > div {
        border-radius: 14px !important;
        min-height: 50px;
    }

    textarea {
        border-radius: 15px !important;
        font-size: 18px !important;
    }

    /* =========================
       ANIMATIONS
       ========================= */

    @keyframes floatCard {
        0%, 100% {
            transform: translateY(0px);
        }

        50% {
            transform: translateY(-5px);
        }
    }

    @keyframes butterflyFloat {
        0%, 100% {
            transform: translateY(0px) rotate(-2deg);
        }

        50% {
            transform: translateY(-8px) rotate(2deg);
        }
    }

    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #f0edff,
            #f8f7ff
        );
    }

    section[data-testid="stSidebar"] p {
        font-size: 17px !important;
    }

    </style>
    """
)


# ============================================================
# HERO SECTION
# ============================================================

render_html(
    """
    <div class="hero">

        <div class="butterfly">🦋</div>

        <div class="hero-title">
            S.L.E.D.
        </div>

        <div class="hero-subtitle">
            Systemic Lupus Erythematosus Detector
        </div>

        <br>

        <div>
            A student-built AI screening prototype
            that combines text, questions and images.
        </div>

    </div>
    """
)


# ============================================================
# INTRODUCTION
# ============================================================

render_html(
    """
    <div class="section-card">

    <h2>✨ How S.L.E.D. Thinks</h2>

    <p>
    S.L.E.D. brings together different types of information to
    create a screening score. It does not diagnose a disease.
    Instead, it looks for patterns in the information provided.
    </p>

    </div>
    """
)


# ============================================================
# WHAT THE TECHNOLOGIES DO
# ============================================================

st.markdown(
    "<h2>🧠 The Technology Behind S.L.E.D.</h2>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    render_html(
        """
        <div class="feature-card">

        <div class="feature-icon">📊</div>

        <div class="feature-title">
        Understanding Data
        </div>

        <div class="feature-text">
        Data Science helps S.L.E.D. organise information,
        identify patterns and turn different inputs into
        useful numerical scores.
        </div>

        </div>
        """
    )

with col2:
    render_html(
        """
        <div class="feature-card">

        <div class="feature-icon">💬</div>

        <div class="feature-title">
        Understanding Human Language
        </div>

        <div class="feature-text">
        Natural Language Processing allows S.L.E.D.
        to read symptom descriptions and understand
        patterns in the words entered by the user.
        </div>

        </div>
        """
    )

with col3:
    render_html(
        """
        <div class="feature-card">

        <div class="feature-icon">📷</div>

        <div class="feature-title">
        Understanding Images
        </div>

        <div class="feature-text">
        Computer Vision allows S.L.E.D. to process
        an uploaded image and identify visual patterns
        learned by the image analysis model.
        </div>

        </div>
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <h2>🦋 S.L.E.D.</h2>

        <p>
        Your AI screening assistant.
        </p>
        """
    )

    st.markdown("---")

    render_html(
        """
        <span class="badge">💬 Text</span>
        <span class="badge">📋 Questions</span>
        <span class="badge">📷 Image</span>
        """
    )

    st.markdown("---")

    st.info(
        "S.L.E.D. is an educational AI prototype. "
        "Its score is not a medical diagnosis."
    )


# ============================================================
# NLP TRAINING DATA
# ============================================================

training_text = [

    "I have joint pain",
    "my joints hurt",
    "my joints are swollen",
    "I have painful swollen joints",

    "I feel extremely tired",
    "I have unusual fatigue",
    "I am constantly exhausted",

    "I get skin rashes",
    "I have a skin rash",
    "my skin develops rashes",

    "sunlight makes my rash worse",
    "I am sensitive to sunlight",

    "I have mouth ulcers",
    "I keep getting sores in my mouth",

    "I have unexplained fever",
    "I keep getting fever",

    "I have unusual hair loss",
    "my hair is falling out",

    "I feel normal",
    "I have no symptoms",
    "I don't have joint pain",
    "I don't have rashes",
    "I don't feel unusually tired",
    "I don't have mouth ulcers",
    "I don't have fever"
]

training_labels = [

    1, 1, 1, 1,

    1, 1, 1,

    1, 1, 1,

    1, 1,

    1, 1,

    1, 1,

    1, 1,

    0, 0, 0, 0, 0, 0, 0
]


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

X_text = vectorizer.fit_transform(training_text)

nlp_model = LogisticRegression(
    max_iter=1000
)

nlp_model.fit(
    X_text,
    training_labels
)

NO_SYMPTOM_TEXTS = {
    "nothing",
    "nothing else",
    "nothing is wrong",
    "nothing wrong",
    "no symptoms",
    "no symptom",
    "i have no symptoms",
    "i have no symptom",
    "i don't have symptoms",
    "i don't have any symptoms",
    "i do not have symptoms",
    "i do not have any symptoms",
    "i feel fine",
    "i am fine",
    "i'm fine",
    "feeling fine",
    "i feel normal",
    "i am normal",
    "i'm okay",
    "i am okay",
    "all good",
    "not experiencing symptoms",
    "not experiencing any symptoms",
    "i don't have joint pain",
    "i don't have rashes",
    "i don't feel unusually tired",
    "i don't have mouth ulcers",
    "i don't have fever",
    "i do not have joint pain",
    "i do not have rashes",
    "i do not feel unusually tired",
    "i do not have mouth ulcers",
    "i do not have fever",
    "i have no joint pain",
    "i have no rashes",
    "i am not unusually tired",
    "i have no mouth ulcers",
    "i have no fever",
}


def analyse_nlp(text):

    normalized_text = re.sub(
        r"[^a-z0-9']+",
        " ",
        text.casefold().replace("’", "'")
    ).strip()

    if not normalized_text or normalized_text in NO_SYMPTOM_TEXTS:
        return 0.0

    vector = vectorizer.transform([text])

    probability = nlp_model.predict_proba(vector)[0][1]

    return probability * 100


# ============================================================
# USER QUESTIONS
# ============================================================

st.markdown(
    "<h2>💬 Tell S.L.E.D. How You're Feeling</h2>",
    unsafe_allow_html=True
)

render_html(
    """
    <div class="section-card">

    <p>
    Answer honestly. These questions are used only as part
    of the educational screening model.
    </p>

    </div>
    """
)


q1 = st.selectbox(
    "😴 Have you been experiencing unusual tiredness or fatigue?",
    ["No", "Sometimes", "Frequently"]
)

q2 = st.selectbox(
    "🦴 Have you experienced joint pain or swelling?",
    ["No", "Sometimes", "Frequently"]
)

q3 = st.selectbox(
    "🌸 Have you noticed unexplained skin rashes?",
    ["No", "Sometimes", "Frequently"]
)

q4 = st.selectbox(
    "☀️ Do sunlight or UV exposure seem to make your skin symptoms worse?",
    ["No", "Sometimes", "Frequently"]
)

q5 = st.selectbox(
    "👄 Have you experienced repeated mouth or nose ulcers?",
    ["No", "Sometimes", "Frequently"]
)

q6 = st.selectbox(
    "🌡️ Have you experienced unexplained fever?",
    ["No", "Sometimes", "Frequently"]
)

q7 = st.selectbox(
    "💇 Have you noticed unusual hair loss?",
    ["No", "Sometimes", "Frequently"]
)


# ============================================================
# ALLERGIES SECTION
# ============================================================

st.markdown(
    "<h2>🌼 Allergies & Other Conditions</h2>",
    unsafe_allow_html=True
)

render_html(
    """
    <div class="section-card">

    <p>
    Allergies can sometimes cause symptoms such as rashes,
    itching or swelling. Tell S.L.E.D. about any known allergies
    so the information can be considered separately from the
    main screening score.
    </p>

    </div>
    """
)


allergy_status = st.selectbox(
    "Do you have any known allergies?",
    [
        "No known allergies",
        "Yes — I have known allergies",
        "I'm not sure"
    ]
)


allergy_details = ""

if allergy_status == "Yes — I have known allergies":

    allergy_details = st.text_area(
        "🌼 What allergies are you aware of?",
        placeholder="Example: pollen, dust, food allergy, medicine allergy...",
        height=110
    )

elif allergy_status == "I'm not sure":

    allergy_details = st.text_area(
        "🌼 Tell S.L.E.D. anything you know about possible allergies.",
        placeholder="Type anything you think might be relevant...",
        height=110
    )


# ============================================================
# FREE TEXT
# ============================================================

st.markdown(
    "<h2>📝 Anything Else?</h2>",
    unsafe_allow_html=True
)

extra_text = st.text_area(
    "Describe anything else you've noticed:",
    placeholder="Example: I have been unusually tired and sometimes get a rash...",
    height=140
)


# ============================================================
# BUILD SYMPTOM TEXT
# ============================================================

symptoms = []

if q1 != "No":
    symptoms.append("unusual tiredness")

if q2 != "No":
    symptoms.append("joint pain")

if q3 != "No":
    symptoms.append("skin rash")

if q4 != "No":
    symptoms.append("sunlight sensitivity")

if q5 != "No":
    symptoms.append("mouth or nose ulcers")

if q6 != "No":
    symptoms.append("unexplained fever")

if q7 != "No":
    symptoms.append("unusual hair loss")


combined_text = " ".join(symptoms)

if extra_text.strip():
    combined_text += " " + extra_text


# ============================================================
# QUESTIONNAIRE SCORE
# ============================================================

def calculate_questionnaire_score():

    answers = [
        q1,
        q2,
        q3,
        q4,
        q5,
        q6,
        q7
    ]

    score = 0

    for answer in answers:

        if answer == "Frequently":
            score += 1

        elif answer == "Sometimes":
            score += 0.5

    return (score / len(answers)) * 100


# ============================================================
# COMPUTER VISION MODEL
# ============================================================

@st.cache_resource
def load_cv_model():
    if not CV_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Teachable Machine model not found: {CV_MODEL_PATH.name}"
        )

    if not CV_LABELS_PATH.is_file():
        raise FileNotFoundError(
            f"Teachable Machine labels not found: {CV_LABELS_PATH.name}"
        )

    label_entries = {}

    for line in CV_LABELS_PATH.read_text(encoding="utf-8").splitlines():
        index_text, separator, label = line.strip().partition(" ")

        if not separator or not index_text.isdigit() or not label.strip():
            raise ValueError(
                f"Invalid class label in {CV_LABELS_PATH.name}: {line!r}"
            )

        index = int(index_text)

        if index in label_entries:
            raise ValueError(
                f"Duplicate class index {index} in {CV_LABELS_PATH.name}"
            )

        label_entries[index] = label.strip()

    if not label_entries or set(label_entries) != set(range(len(label_entries))):
        raise ValueError(
            f"Class indices in {CV_LABELS_PATH.name} must start at 0 "
            "and be consecutive."
        )

    class_names = [
        label_entries[index]
        for index in range(len(label_entries))
    ]

    model = tf.keras.models.load_model(
        CV_MODEL_PATH,
        compile=False
    )

    if len(model.input_shape) != 4 or model.input_shape[-1] != 3:
        raise ValueError(
            "The Teachable Machine model must accept RGB images."
        )

    if len(model.output_shape) != 2 or model.output_shape[-1] != len(class_names):
        raise ValueError(
            "The number of model outputs does not match the classes in "
            f"{CV_LABELS_PATH.name}."
        )

    if not any(
        label.casefold() == "lupus"
        for label in class_names
    ):
        raise ValueError(
            f"No 'Lupus' class was found in {CV_LABELS_PATH.name}."
        )

    return model, class_names


def predict_image(model, image):
    input_height, input_width = model.input_shape[1:3]

    if input_height is None or input_width is None:
        raise ValueError(
            "The Teachable Machine model must have fixed image dimensions."
        )

    image = image.resize(
        (input_width, input_height)
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = (array / 127.5) - 1.0
    array = np.expand_dims(
        array,
        axis=0
    )

    return model.predict(
        array,
        verbose=0
    )[0]


# ============================================================
# LOAD PRE-TRAINED IMAGE MODEL
# ============================================================

st.markdown(
    "<h2>🧠 How Image Analysis Works</h2>",
    unsafe_allow_html=True
)

render_html(
    """
    <div class="section-card">

    <p>
    When you upload an image, S.L.E.D. looks for visual patterns
    associated with its Lupus and Non-Lupus categories. The image
    score shows how strongly the image matches the Lupus category
    in this prototype. It is not a diagnosis, and an image alone
    cannot determine whether someone has lupus.
    </p>

    </div>
    """
)

cv_model = None
class_names = []

if CV_MODEL_PATH.is_file() and CV_LABELS_PATH.is_file():
    cv_model, class_names = load_cv_model()
else:
    st.warning(
        "Image analysis is currently unavailable."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    "<h2>📷 Let S.L.E.D. Look at an Image</h2>",
    unsafe_allow_html=True
)

uploaded_image = st.file_uploader(
    "Upload an image for the Computer Vision model:",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded_image is not None:

    image = Image.open(
        uploaded_image
    ).convert("RGB")

    st.image(
        image,
        caption="Your uploaded image",
        width=350
    )


# ============================================================
# ANALYSIS BUTTON
# ============================================================

st.markdown(
    "<h2>✨ Ready?</h2>",
    unsafe_allow_html=True
)

render_html(
    """
    <div class="section-card">

    <p>
    S.L.E.D. will combine the information you provided
    and create a screening score.
    </p>

    </div>
    """
)


if st.button(
    "🦋 RUN S.L.E.D. ANALYSIS"
):

    with st.spinner(
        "S.L.E.D. is analysing your information... ✨"
    ):

        # --------------------------------
        # NLP
        # --------------------------------

        nlp_score = analyse_nlp(
            combined_text
        )

        # --------------------------------
        # QUESTIONNAIRE
        # --------------------------------

        questionnaire_score = (
            calculate_questionnaire_score()
        )

        # --------------------------------
        # COMPUTER VISION
        # --------------------------------

        cv_lupus_score = 0.0

        if uploaded_image is not None:

            if cv_model is not None:

                class_scores = predict_image(
                    cv_model,
                    image
                )

                lupus_index = next(
                    index
                    for index, label in enumerate(class_names)
                    if label.casefold() == "lupus"
                )

                cv_lupus_score = float(
                    class_scores[lupus_index] * 100
                )

            else:

                st.warning(
                    "Image analysis is currently unavailable. "
                    "The image component will not be included."
                )

        else:

            st.info(
                "No image was uploaded, so the Computer Vision "
                "component is not being used."
            )

        # --------------------------------
        # FINAL SCORE
        # --------------------------------

        if uploaded_image is not None and cv_model is not None:

            final_score = (

                nlp_score * 0.30

                +

                questionnaire_score * 0.30

                +

                cv_lupus_score * 0.40
            )

        else:

            final_score = (

                nlp_score * 0.50

                +

                questionnaire_score * 0.50
            )


        # --------------------------------
        # RESULT CATEGORY
        # --------------------------------

        if final_score < 35:

            category = "LOWER SCREENING INDICATION"

            emoji = "🌱"

            message = (
                "The information provided shows a lower "
                "screening indication in this prototype."
            )

            recommendation_points = [
                "A low prototype score cannot rule out lupus or explain symptoms. "
                "If you have symptoms or concerns, a doctor can assess you and "
                "consider other possible causes, such as allergies, rosacea, or "
                "vitamin deficiencies.",
                "Seek medical advice if new symptoms appear or existing symptoms "
                "persist or worsen, including a rash that worsens after sun exposure.",
            ]

        elif final_score < 65:

            category = "MODERATE SCREENING INDICATION"

            emoji = "🌼"

            message = (
                "The information provided shows a moderate "
                "screening indication in this prototype."
            )

            recommendation_points = [
                "If symptoms persist or concern you, arrange an appointment with "
                "a general physician for an evaluation.",
                "Keep a symptom journal. You can repeat this checklist in 2–4 "
                "weeks if symptoms remain stable, but seek care sooner if they "
                "worsen.",
                "For general wellbeing, consider broad-spectrum SPF 50+ sunscreen "
                "and protective clothing if sunlight aggravates your symptoms, "
                "along with sufficient sleep, gentle activity as tolerated, and "
                "a balanced diet.",
            ]

        else:

            category = "HIGHER SCREENING INDICATION"

            emoji = "🦋"

            message = (
                "The information provided shows a higher "
                "screening indication in this prototype."
            )

            recommendation_points = [
                "This prototype score is not a diagnosis or a medical probability. "
                "Arrange a timely appointment with a doctor to discuss your "
                "symptoms; ask whether evaluation by a rheumatologist is appropriate.",
                "Your doctor can decide whether tests such as anti-dsDNA, "
                "anti-Sm, complement C3/C4, or urine protein testing are relevant "
                "for you.",
                "Keep a daily symptom journal, including symptoms such as joint "
                "pain, temperature, and fatigue, to discuss at your appointment.",
                "Seek emergency care for severe or sudden symptoms such as chest "
                "pain, shortness of breath, or sudden severe swelling in the legs.",
            ]


        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown(
            "<h2>✨ Your S.L.E.D. Results</h2>",
            unsafe_allow_html=True
        )

        render_html(
            f"""
            <div class="result-card">

                <div style="font-size:55px;">
                    {emoji}
                </div>

                <div class="result-label">
                    SCREENING SCORE
                </div>

                <div class="result-number">
                    {final_score:.1f}%
                </div>

                <div class="result-label">
                    {category}
                </div>

                <br>

                <p>
                    {message}
                </p>

            </div>
            """
        )

        st.markdown(
            "### ✅ What to consider next"
        )

        st.markdown(
            "\n".join(
                f"- {point}"
                for point in recommendation_points
            )
        )


        # ====================================================
        # BREAKDOWN
        # ====================================================

        st.markdown(
            "<h2>📊 Where Did The Score Come From?</h2>",
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "💬 Text Analysis",
                f"{nlp_score:.1f}%"
            )

        with c2:

            st.metric(
                "📋 Question Analysis",
                f"{questionnaire_score:.1f}%"
            )

        with c3:

            if uploaded_image is not None and cv_model is not None:

                st.metric(
                    "📷 Image Analysis",
                    f"{cv_lupus_score:.1f}%"
                )

            else:

                st.metric(
                    "📷 Image Analysis",
                    "Not used"
                )


        # ====================================================
        # ALLERGY INFORMATION
        # ====================================================

        st.markdown(
            "<h2>🌼 Allergy Information</h2>",
            unsafe_allow_html=True
        )

        if allergy_status == "Yes — I have known allergies":

            st.info(
                "You reported having known allergies. "
                "Allergies can cause symptoms that may overlap "
                "with some skin or other symptoms, so a healthcare "
                "professional should consider your complete history."
            )

            if allergy_details.strip():

                st.write(
                    "**Allergy information provided:**"
                )

                st.write(
                    allergy_details
                )

        elif allergy_status == "I'm not sure":

            st.info(
                "You indicated that you are unsure about possible "
                "allergies. A doctor can help determine whether "
                "your symptoms could have an allergic cause."
            )

        else:

            st.success(
                "No known allergies were reported."
            )


        # ====================================================
        # S.L.E.D. BOT
        # ====================================================

        st.markdown(
            "<h2>🤖 S.L.E.D. Says...</h2>",
            unsafe_allow_html=True
        )

        render_html(
            f"""
            <div class="section-card">

            <p>
            Hey! 🦋 I've finished analysing the information
            you gave me.
            </p>

            <p>
            My prototype screening score is
            <strong>{final_score:.1f}%</strong>.
            </p>

            <p>
            Remember: this number represents the output of
            a student-built AI model. It is <strong>not</strong>
            a medical probability or diagnosis.
            </p>

            </div>
            """
        )


        # ====================================================
        # DOCTOR MESSAGE
        # ====================================================

        render_html(
            """
            <div class="doctor-card">

                <div class="doctor-title">
                    🩺 One Last Thing
                </div>

                <p class="doctor-text">

                Please don't use S.L.E.D. to diagnose yourself.
                If you are experiencing symptoms or are concerned
                about your health, <strong>please visit a qualified
                doctor or healthcare professional.</strong>

                <br><br>

                A doctor can consider your symptoms, medical history
                and appropriate clinical tests to give you reliable
                advice.

                <br><br>

                💙 <strong>S.L.E.D. can screen. A doctor can diagnose.</strong>

                </p>

            </div>
            """
        )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <br><br>

    <div style="
        text-align:center;
        padding:25px;
        color:#666;
        font-size:16px;
    ">

    🦋 <strong>S.L.E.D.</strong>
    <br>
    A Class 12 Artificial Intelligence Project
    <br><br>

    Built using Python • Streamlit • Machine Learning •
    Natural Language Processing • Computer Vision

    <br><br>

    <em>
    Educational prototype — not a medical diagnostic tool.
    </em>

    </div>
    """
)