mkdir -p ~/.streamlit/
python3 -m spacy download en_core_web_md
echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml