import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Show title and description.
st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "fr_XX"

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

encoded_p = tokenizer(sentence, return_tensors="pt")
generated_tokens = model.generate(**encoded_p, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
tgt_translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

response = f"Translate fr to en: {*tgt_translation}"
# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response)
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})
