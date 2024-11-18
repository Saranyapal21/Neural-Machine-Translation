import os
import time
import pickle
import streamlit as st
import textwrap
import html


st.set_page_config(page_title="Neural Machine Translation", page_icon="google-translate.png", layout="centered")


def load_css(filepath: str):
    with open(filepath, "r") as f:
        return f.read()


def wrap_and_escape_text(text, width=100):
    escaped_text = html.escape(text)
    wrapped_lines = textwrap.wrap(escaped_text, width=width)
    return '<br>'.join(wrapped_lines)


def load_model():
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer


def translate_text(model, tokenizer, text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded_text = tokenizer(text, return_tensors="pt")
    forced_bos_token_id = tokenizer.get_lang_id(tgt_lang)
    generated_tokens = model.generate(
        **encoded_text,
        forced_bos_token_id=forced_bos_token_id,
        max_length=70
    )
    translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translation


pair_to_lang = {
    'English to Bangla': ('en', 'bn'),
    'English to Hindi': ('en', 'hi'),
    'Bangla to English': ('bn', 'en'),
    'Hindi to English': ('hi', 'en')
}


css_path = os.path.join(os.path.dirname(__file__), "style.css")  # Ensure the correct path
css = load_css(css_path)
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


if 'translations' not in st.session_state:
    st.session_state['translations'] = []


if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown('<h4 style="color:white; font-weight:bold;">Loading Translation Model...</h4>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        for percent_complete in range(0, 100, 5):
            time.sleep(6)
            progress_bar.progress(percent_complete + 5)

        
        model, tokenizer = load_model()
        st.session_state['model'] = model
        st.session_state['tokenizer'] = tokenizer

    
    loading_placeholder.empty()
    st.success('Model loaded successfully!')
else:
    model = st.session_state['model']
    tokenizer = st.session_state['tokenizer']


st.markdown('<div class="typing-title">Neural Machine Translation</div>', unsafe_allow_html=True)
time.sleep(1)  


st.markdown('<h3>Translate between English, Hindi, and Bengali</h3>', unsafe_allow_html=True)


with st.form(key='translation_form', clear_on_submit=True):
    language_pairs = list(pair_to_lang.keys())
    selected_pair = st.selectbox("Select Translation Direction", language_pairs, key='selected_pair')

    input_text = st.text_area("Enter Text to Translate", height=150, key='form_input_text')

    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if input_text.strip() == "":
            st.warning("Please enter text to translate.")
        else:
            loading_placeholder = st.empty()
            with loading_placeholder:
                st.markdown('<div class="custom-loader"></div>', unsafe_allow_html=True)

            src_lang, tgt_lang = pair_to_lang[selected_pair]
            translated_output = translate_text(model, tokenizer, input_text, src_lang, tgt_lang)
            loading_placeholder.empty()

            
            st.session_state['translations'].append({
                "input": input_text,
                "output": translated_output,
                "pair": selected_pair
            })


            unique_id = str(int(time.time() * 1000))
            wrapped_output = wrap_and_escape_text(translated_output)
            st.markdown(f'''
            <style>
            @keyframes hideCheckmark_{unique_id} {{
                0% {{ opacity: 1; }}
                100% {{ opacity: 0; }}
            }}
            </style>
            <div class="translation-completed">
                <div class="checkmark" style="animation: hideCheckmark_{unique_id} 1.5s forwards;">✓</div>
                <div class="output-content">
                    <div class="output-label">Output:</div>
                    <div class="output-bar">{wrapped_output}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)


if st.session_state['translations']:
    st.markdown("### Translation History")
    for idx, translation in enumerate(st.session_state['translations'], 1):
        wrapped_input = wrap_and_escape_text(translation['input'])
        wrapped_output = wrap_and_escape_text(translation['output'])
        st.markdown(f'''
        <div class="translated-text">
            <strong>#{idx} {translation['pair']}</strong><br>
            <em>Input:</em> {wrapped_input}<br>
            <em>Output:</em> {wrapped_output}
        </div>
        ''', unsafe_allow_html=True)

st.markdown('<div class="footer">© 2024 Neural Machine Translation App</div>', unsafe_allow_html=True)