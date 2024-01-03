FROM python:3.11.7
RUN apt update -y
COPY . /usr
WORKDIR /usr
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ['streamlit', 'run', 'streamlit.py']


