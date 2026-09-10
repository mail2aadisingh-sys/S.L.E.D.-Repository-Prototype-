import streamlit as st
import numpy as np
import os
import zipfile
import tempfile

from PIL import Image

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# CV
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# S.L.E.D.
# SYSTEMIC LUPUS ERYTHEMATOSUS DETECTOR
# ============================================================

st.set_page_config(
    page_title="S.L.E.D.",
    page_icon="🦋",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

LUPUS_ZIP = "Lupus Training Data.zip"
NON_LUPUS_ZIP = "Non-Lupus Training Data.zip"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "keras_model.h5"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "labels.txt"
)

IMAGE_SIZE = (160, 160)
TEACHABLE_MACHINE_IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #09090d;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

.sled-title {
    text-align: center;
    font-size: 64px;
    font-weight: 800;
    letter-spacing: -4px;
}

.sled-subtitle {
    text-align: center;
    color: #aaaaba;
    font-size: 19px;
}

.sled-tagline {
    text-align: center;
    color: #777789;
    margin-bottom: 35px;
}

.card {
    background: #17171f;
    border: 1px solid #2d2d38;
    border-radius: 22px;
    padding: 25px;
    margin-bottom: 22px;
}

.bot {
    background: #1e1a2b;
    border-left: 4px solid #9b7cff;
    border-radius: 18px;
    padding: 20px;
    margin: 20px 0;
}

.result {
    background: #17171f;
    border: 1px solid #393945;
    border-radius: 25px;
    padding: 35px;
    text-align: center;
}

.big-score {
    font-size: 50px;
    font-weight: 800;
}

.small {
    color: #888895;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="sled-title">S.L.E.D. 🦋</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sled-subtitle">'
    'Systemic Lupus Erythematosus Detector'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sled-tagline">'
    'Natural Language Processing • Computer Vision • Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# BOT
# ============================================================

st.markdown("""
<div class="bot">

<b>🦋 S.L.E.D. AI</b>

<br><br>

Hi! I'm S.L.E.D. 👋

<br><br>

Systemic Lupus Erythematosus, also known as <b>SLE</b>, is a chronic
autoimmune disease where the body's own cells mistakenly attack the body.
It can affect several vital systems, including the joints, skin, kidneys,
blood cells, brain, heart and lungs. Unfortunately, SLE is often misdiagnosed
because its symptoms can be common and difficult to identify.

<br><br>

Fortunately, I, S.L.E.D., can help with an initial screening!

<br><br>

I'll analyse your answers using <b>NLP</b> and, if you
upload an image, I'll use a <b>Computer Vision model</b>
trained using lupus and non-lupus image datasets.

<br><br>

I'll then combine the results and give you a
percentage-based <b>screening score</b>.

<br><br>

<span class="small">
Educational prototype — not a medical diagnosis.
</span>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🦋 S.L.E.D.")

    st.write(
        "Systemic Lupus Erythematosus Detector"
    )

    st.divider()

    st.write("💬 NLP")
    st.write("📷 Computer Vision")
    st.write("🧠 MobileNetV2")
    st.write("📊 AI Score Fusion")

    st.divider()

    st.info(
        "The percentages produced by S.L.E.D. "
        "represent model scores/confidence, not "
        "a medical probability."
    )


# ============================================================
# ============================================================
# NLP SECTION
# ============================================================
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.header("💬 Step 1 — Talk to S.L.E.D.")

st.write(
    "Answer the questions below. You can also describe "
    "your symptoms in your own words."
)


q1 = st.radio(
    "Do you frequently experience unexplained tiredness?",
    ["No", "Sometimes", "Frequently"],
    horizontal=True
)

q2 = st.radio(
    "Have you experienced recurring joint pain or swelling?",
    ["No", "Sometimes", "Frequently"],
    horizontal=True
)

q3 = st.radio(
    "Have you noticed recurring skin rashes?",
    ["No", "Sometimes", "Frequently"],
    horizontal=True
)

q4 = st.radio(
    "Does sunlight seem to make your skin symptoms worse?",
    ["No", "Sometimes", "Frequently"],
    horizontal=True
)

q5 = st.radio(
    "Do you frequently experience mouth or nose ulcers?",
    ["No", "Sometimes", "Frequently"],
    horizontal=True
)

q6 = st.radio(
    "Have you experienced unexplained fever?",
    ["No", "Sometimes", "Frequently"],
    horizontal=True
)

q7 = st.radio(
    "Have you noticed unusual hair loss?",
    ["No", "Sometimes", "Frequently"],
    horizontal=True
)

extra_text = st.text_area(
    "📝 Describe anything else:",
    placeholder="Example: I sometimes have joint pain and feel unusually tired..."
)

st.markdown("</div>", unsafe_allow_html=True)


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

    0, 0, 0, 0,
    0, 0, 0
]


# ============================================================
# TRAIN NLP
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

X_text = vectorizer.fit_transform(
    training_text
)

nlp_model = LogisticRegression(
    max_iter=1000
)

nlp_model.fit(
    X_text,
    training_labels
)


# ============================================================
# NLP ANALYSIS
# ============================================================

def analyse_nlp(text):

    vector = vectorizer.transform(
        [text]
    )

    probability = nlp_model.predict_proba(
        vector
    )[0][1]

    return probability * 100


# ============================================================
# CREATE NLP INPUT
# ============================================================

symptoms = []

if q1 != "No":
    symptoms.append("I have unusual fatigue")

if q2 != "No":
    symptoms.append("I have joint pain and swelling")

if q3 != "No":
    symptoms.append("I have skin rashes")

if q4 != "No":
    symptoms.append("sunlight makes my rash worse")

if q5 != "No":
    symptoms.append("I have mouth ulcers")

if q6 != "No":
    symptoms.append("I have unexplained fever")

if q7 != "No":
    symptoms.append("I have unusual hair loss")

if extra_text.strip():
    symptoms.append(extra_text)

combined_text = " ".join(symptoms)


# ============================================================
# QUESTIONNAIRE SCORE
# ============================================================

def calculate_questionnaire_score():

    answers = [
        q1, q2, q3, q4,
        q5, q6, q7
    ]

    score = 0

    for answer in answers:

        if answer == "Frequently":
            score += 1

        elif answer == "Sometimes":
            score += 0.5

    return (
        score / len(answers)
    ) * 100


# ============================================================
# ============================================================
# COMPUTER VISION
# ============================================================
# ============================================================


def extract_zip(zip_path, destination):

    os.makedirs(
        destination,
        exist_ok=True
    )

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as z:

        z.extractall(
            destination
        )


# ============================================================
# PREPARE DATASET
# ============================================================

@st.cache_resource
def prepare_dataset():

    if not os.path.exists(LUPUS_ZIP):
        return None

    if not os.path.exists(NON_LUPUS_ZIP):
        return None

    base_dir = tempfile.mkdtemp(
        prefix="sled_dataset_"
    )

    lupus_dir = os.path.join(
        base_dir,
        "lupus"
    )

    non_lupus_dir = os.path.join(
        base_dir,
        "non_lupus"
    )

    extract_zip(
        LUPUS_ZIP,
        lupus_dir
    )

    extract_zip(
        NON_LUPUS_ZIP,
        non_lupus_dir
    )

    # Create clean directory
    clean_dir = os.path.join(
        base_dir,
        "clean"
    )

    train_dir = os.path.join(
        clean_dir,
        "train"
    )

    val_dir = os.path.join(
        clean_dir,
        "validation"
    )

    os.makedirs(train_dir)
    os.makedirs(val_dir)

    for category in [
        "lupus",
        "non_lupus"
    ]:

        os.makedirs(
            os.path.join(
                train_dir,
                category
            )
        )

        os.makedirs(
            os.path.join(
                val_dir,
                category
            )
        )

    # Collect images
    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    )

    lupus_images = []

    non_lupus_images = []

    for root, dirs, files in os.walk(
        lupus_dir
    ):

        for file in files:

            if file.lower().endswith(
                image_extensions
            ):

                lupus_images.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    for root, dirs, files in os.walk(
        non_lupus_dir
    ):

        for file in files:

            if file.lower().endswith(
                image_extensions
            ):

                non_lupus_images.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    # Shuffle
    rng = np.random.default_rng(
        42
    )

    rng.shuffle(
        lupus_images
    )

    rng.shuffle(
        non_lupus_images
    )

    # 80/20 split
    lupus_split = int(
        len(lupus_images) * 0.8
    )

    non_lupus_split = int(
        len(non_lupus_images) * 0.8
    )

    datasets = {

        "lupus": (
            lupus_images[:lupus_split],
            lupus_images[lupus_split:]
        ),

        "non_lupus": (
            non_lupus_images[:non_lupus_split],
            non_lupus_images[non_lupus_split:]
        )
    }

    for category, (
        train_images,
        validation_images
    ) in datasets.items():

        for index, source in enumerate(
            train_images
        ):

            destination = os.path.join(
                train_dir,
                category,
                f"{index}.jpg"
            )

            try:

                image = Image.open(
                    source
                ).convert("RGB")

                image.save(
                    destination,
                    "JPEG"
                )

            except:
                pass

        for index, source in enumerate(
            validation_images
        ):

            destination = os.path.join(
                val_dir,
                category,
                f"{index}.jpg"
            )

            try:

                image = Image.open(
                    source
                ).convert("RGB")

                image.save(
                    destination,
                    "JPEG"
                )

            except:
                pass

    return clean_dir


# ============================================================
# BUILD CV MODEL
# ============================================================

@st.cache_resource
def build_cv_model(dataset_dir):

    train_dir = os.path.join(
        dataset_dir,
        "train"
    )

    val_dir = os.path.join(
        dataset_dir,
        "validation"
    )

    train_data = tf.keras.utils.image_dataset_from_directory(

        train_dir,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="binary",

        shuffle=True,

        seed=42
    )

    validation_data = tf.keras.utils.image_dataset_from_directory(

        val_dir,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="binary",

        shuffle=False
    )

    class_names = train_data.class_names

    # Performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_data = train_data.prefetch(
        AUTOTUNE
    )

    validation_data = validation_data.prefetch(
        AUTOTUNE
    )

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    augmentation = tf.keras.Sequential([

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.05
        ),

        layers.RandomZoom(
            0.10
        )
    ])


    # --------------------------------------------------------
    # MobileNetV2
    # --------------------------------------------------------

    base_model = MobileNetV2(

        input_shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ),

        include_top=False,

        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = False


    # --------------------------------------------------------
    # S.L.E.D. model
    # --------------------------------------------------------

    inputs = layers.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    )

    x = augmentation(
        inputs
    )

    x = preprocess_input(
        x
    )

    x = base_model(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(
        x
    )

    x = layers.Dropout(
        0.30
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )


    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = model.fit(

        train_data,

        validation_data=validation_data,

        epochs=EPOCHS,

        verbose=0
    )

    return (
        model,
        class_names,
        history
    )


# ============================================================
# LOAD CV MODEL
# ============================================================

@st.cache_resource
def load_cv_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        labels = [
            line.strip().split(maxsplit=1)[-1]
            for line in file
            if line.strip()
        ]

    return model, labels


cv_model = None
labels = []

if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):

    try:
        cv_model, labels = load_cv_model()

        st.success(
            "✓ Computer Vision model loaded"
        )

    except Exception as error:

        st.error(
            "The Teachable Machine model could not be loaded. "
            "Check that keras_model.h5 and labels.txt are valid."
        )
        st.caption(str(error))

else:

    st.warning(
        "Add keras_model.h5 and labels.txt beside this Python file "
        "to enable image analysis."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.header(
    "📷 Step 2 — Computer Vision"
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)

uploaded_image = None

if uploaded_file:

    uploaded_image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        uploaded_image,
        caption="Image submitted to S.L.E.D.",
        width=400
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# CV PREDICTION
# ============================================================

def predict_image(
    model,
    image
):

    image = image.resize(
        TEACHABLE_MACHINE_IMAGE_SIZE
    )

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    array = np.expand_dims(
        array,
        axis=0
    )

    predictions = model.predict(
        array,
        verbose=0
    )[0]

    return predictions


# ============================================================
# RUN S.L.E.D.
# ============================================================

if st.button(
    "🦋 RUN S.L.E.D. ANALYSIS",
    use_container_width=True
):

    if not combined_text.strip():

        st.warning(
            "Please answer the questions first."
        )

    elif (
        uploaded_image is not None
        and cv_model is None
    ):

        st.warning(
            "The Computer Vision model is not available."
        )

    else:

        # ====================================================
        # NLP
        # ====================================================

        nlp_score = analyse_nlp(
            combined_text
        )


        # ====================================================
        # QUESTIONNAIRE
        # ====================================================

        questionnaire_score = (
            calculate_questionnaire_score()
        )


        # ====================================================
        # CV
        # ====================================================

        cv_lupus_score = None
        cv_non_lupus_score = None

        if uploaded_image is not None:

            predictions = predict_image(
                cv_model,
                uploaded_image
            )

            lupus_index = next(
                (
                    index for index, label in enumerate(labels)
                    if "lupus" in label.lower()
                    and "non" not in label.lower()
                ),
                None
            )

            if lupus_index is None or lupus_index >= len(predictions):

                st.error(
                    "labels.txt must contain a label named lupus."
                )
                st.stop()

            cv_lupus_score = float(
                predictions[lupus_index] * 100
            )

            cv_non_lupus_score = (
                100 - cv_lupus_score
            )


        # ====================================================
        # FINAL SCORE
        # ====================================================

        if cv_lupus_score is not None:

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


        final_score = float(
            np.clip(
                final_score,
                0,
                100
            )
        )


        # ====================================================
        # CATEGORY
        # ====================================================

        if final_score < 35:

            category = (
                "LOWER SCREENING INDICATION"
            )

            emoji = "🌱"

        elif final_score < 65:

            category = (
                "MODERATE SCREENING INDICATION"
            )

            emoji = "🟡"

        else:

            category = (
                "HIGHER SCREENING INDICATION"
            )

            emoji = "⚠️"


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="result">',
            unsafe_allow_html=True
        )

        st.subheader(
            "🦋 S.L.E.D. ANALYSIS COMPLETE"
        )

        st.markdown(
            f"### {emoji} {category}"
        )

        st.markdown(
            f'<div class="big-score">'
            f'{final_score:.1f}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Overall S.L.E.D. screening score"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # ====================================================
        # BREAKDOWN
        # ====================================================

        st.subheader(
            "📊 S.L.E.D. Score Breakdown"
        )

        if cv_lupus_score is not None:

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "💬 NLP",
                    f"{nlp_score:.1f}%"
                )

            with c2:

                st.metric(
                    "📋 Questions",
                    f"{questionnaire_score:.1f}%"
                )

            with c3:

                st.metric(
                    "📷 CV — Lupus",
                    f"{cv_lupus_score:.1f}%"
                )

            with c4:

                st.metric(
                    "🦋 S.L.E.D.",
                    f"{final_score:.1f}%"
                )


            st.subheader(
                "📷 Computer Vision Result"
            )

            st.progress(
                int(
                    cv_lupus_score
                )
            )

            st.write(
                f"🦋 Lupus: **{cv_lupus_score:.1f}%**"
            )

            st.write(
                f"✅ Non-lupus: **{cv_non_lupus_score:.1f}%**"
            )

        else:

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "💬 NLP",
                    f"{nlp_score:.1f}%"
                )

            with c2:

                st.metric(
                    "📋 Questions",
                    f"{questionnaire_score:.1f}%"
                )

            with c3:

                st.metric(
                    "🦋 S.L.E.D.",
                    f"{final_score:.1f}%"
                )


        # ====================================================
        # BOT
        # ====================================================

        st.markdown("""
        <div class="bot">

        <b>🦋 S.L.E.D. AI</b>

        <br><br>

        I've finished processing your information.

        The NLP component analysed the text, the questionnaire
        converted the answers into a numerical score, and the
        Computer Vision model compared the uploaded image
        against the two classes it was trained on.

        <br><br>

        <b>Remember:</b> these percentages are model outputs
        from an educational prototype. They are not a medical
        diagnosis or a clinically validated probability.

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="text-align:center;color:#777789;">

🦋 <b>S.L.E.D.</b><br>

Systemic Lupus Erythematosus Detector

<br><br>

NLP • Computer Vision • MobileNetV2 • Machine Learning

<br><br>

Educational Prototype — Not a Medical Diagnostic System

</div>
""", unsafe_allow_html=True)