mkdir -p ~/.streamlit/
python3 -m spacy download en_core_web_md

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml

python3 -m nltk.downloader 'punkt'
python3 -m nltk.downloader 'averaged_perceptron_tagger'
python3 -m nltk.downloader 'wordnet'