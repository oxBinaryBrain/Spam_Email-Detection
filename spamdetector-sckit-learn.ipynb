{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7a2b9f4b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-01T09:57:08.642638Z",
     "iopub.status.busy": "2024-03-01T09:57:08.642263Z",
     "iopub.status.idle": "2024-03-01T09:57:11.108356Z",
     "shell.execute_reply": "2024-03-01T09:57:11.107000Z"
    },
    "papermill": {
     "duration": 2.474027,
     "end_time": "2024-03-01T09:57:11.111204",
     "exception": false,
     "start_time": "2024-03-01T09:57:08.637177",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1e4f1304",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-01T09:57:11.119007Z",
     "iopub.status.busy": "2024-03-01T09:57:11.118577Z",
     "iopub.status.idle": "2024-03-01T09:57:11.304464Z",
     "shell.execute_reply": "2024-03-01T09:57:11.303289Z"
    },
    "papermill": {
     "duration": 0.192755,
     "end_time": "2024-03-01T09:57:11.307083",
     "exception": false,
     "start_time": "2024-03-01T09:57:11.114328",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "# Load the dataset from CSV file\n",
    "data = pd.read_csv('//kaggle/input/spam-email-dataset/emails.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ebe5b035",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-01T09:57:11.314906Z",
     "iopub.status.busy": "2024-03-01T09:57:11.314562Z",
     "iopub.status.idle": "2024-03-01T09:57:11.320372Z",
     "shell.execute_reply": "2024-03-01T09:57:11.319074Z"
    },
    "papermill": {
     "duration": 0.012188,
     "end_time": "2024-03-01T09:57:11.322494",
     "exception": false,
     "start_time": "2024-03-01T09:57:11.310306",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['text', 'spam'], dtype='object')\n"
     ]
    }
   ],
   "source": [
    "# Inspect column names\n",
    "print(data.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "214e932b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-01T09:57:11.330975Z",
     "iopub.status.busy": "2024-03-01T09:57:11.330180Z",
     "iopub.status.idle": "2024-03-01T09:57:11.337994Z",
     "shell.execute_reply": "2024-03-01T09:57:11.337240Z"
    },
    "papermill": {
     "duration": 0.014594,
     "end_time": "2024-03-01T09:57:11.340228",
     "exception": false,
     "start_time": "2024-03-01T09:57:11.325634",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Assuming your CSV has two columns: 'text' for email content and 'spam' for spam or not spam\n",
    "X = data['text']\n",
    "y = data['spam']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "b496ddaf",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-01T09:57:11.347875Z",
     "iopub.status.busy": "2024-03-01T09:57:11.347332Z",
     "iopub.status.idle": "2024-03-01T09:57:11.358983Z",
     "shell.execute_reply": "2024-03-01T09:57:11.358066Z"
    },
    "papermill": {
     "duration": 0.017877,
     "end_time": "2024-03-01T09:57:11.361280",
     "exception": false,
     "start_time": "2024-03-01T09:57:11.343403",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Split the data into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8236beb9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-01T09:57:11.369297Z",
     "iopub.status.busy": "2024-03-01T09:57:11.368778Z",
     "iopub.status.idle": "2024-03-01T09:57:12.375307Z",
     "shell.execute_reply": "2024-03-01T09:57:12.374124Z"
    },
    "papermill": {
     "duration": 1.012955,
     "end_time": "2024-03-01T09:57:12.377502",
     "exception": false,
     "start_time": "2024-03-01T09:57:11.364547",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Vectorize the emails using Bag-of-Words representation\n",
    "vectorizer = CountVectorizer()\n",
    "X_train_counts = vectorizer.fit_transform(X_train)\n",
    "X_test_counts = vectorizer.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1912c69a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-01T09:57:12.386795Z",
     "iopub.status.busy": "2024-03-01T09:57:12.386426Z",
     "iopub.status.idle": "2024-03-01T09:57:12.408324Z",
     "shell.execute_reply": "2024-03-01T09:57:12.407381Z"
    },
    "papermill": {
     "duration": 0.028399,
     "end_time": "2024-03-01T09:57:12.410180",
     "exception": false,
     "start_time": "2024-03-01T09:57:12.381781",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MultinomialNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MultinomialNB</label><div class=\"sk-toggleable__content\"><pre>MultinomialNB()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "MultinomialNB()"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Train a Naive Bayes classifier\n",
    "clf = MultinomialNB()\n",
    "clf.fit(X_train_counts, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "6ebbc303",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-01T09:57:12.417762Z",
     "iopub.status.busy": "2024-03-01T09:57:12.417439Z",
     "iopub.status.idle": "2024-03-01T09:57:12.425851Z",
     "shell.execute_reply": "2024-03-01T09:57:12.424374Z"
    },
    "papermill": {
     "duration": 0.015295,
     "end_time": "2024-03-01T09:57:12.428624",
     "exception": false,
     "start_time": "2024-03-01T09:57:12.413329",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Predictions\n",
    "predictions = clf.predict(X_test_counts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "00be0955",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-01T09:57:12.436969Z",
     "iopub.status.busy": "2024-03-01T09:57:12.436601Z",
     "iopub.status.idle": "2024-03-01T09:57:12.443941Z",
     "shell.execute_reply": "2024-03-01T09:57:12.442815Z"
    },
    "papermill": {
     "duration": 0.013891,
     "end_time": "2024-03-01T09:57:12.445904",
     "exception": false,
     "start_time": "2024-03-01T09:57:12.432013",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.987783595113438\n"
     ]
    }
   ],
   "source": [
    "# Evaluate the model\n",
    "accuracy = accuracy_score(y_test, predictions)\n",
    "print(\"Accuracy:\", accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "c1ccca54",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-03-01T09:57:12.453329Z",
     "iopub.status.busy": "2024-03-01T09:57:12.452994Z",
     "iopub.status.idle": "2024-03-01T09:57:12.461048Z",
     "shell.execute_reply": "2024-03-01T09:57:12.459463Z"
    },
    "papermill": {
     "duration": 0.014392,
     "end_time": "2024-03-01T09:57:12.463463",
     "exception": false,
     "start_time": "2024-03-01T09:57:12.449071",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "'Free offer! Click now to win a holiday.' is classified as SPAM.\n",
      "'Meeting rescheduled to tomorrow.' is classified as NOT SPAM.\n"
     ]
    }
   ],
   "source": [
    "# Test with new emails\n",
    "new_emails = [\n",
    "    \"Free offer! Click now to win a holiday.\",\n",
    "    \"Meeting rescheduled to tomorrow.\",\n",
    "]\n",
    "new_emails_counts = vectorizer.transform(new_emails)\n",
    "new_predictions = clf.predict(new_emails_counts)\n",
    "\n",
    "for email, prediction in zip(new_emails, new_predictions):\n",
    "    if prediction == 1:\n",
    "        print(f\"'{email}' is classified as SPAM.\")\n",
    "    else:\n",
    "        print(f\"'{email}' is classified as NOT SPAM.\")"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 3690036,
     "sourceId": 6399975,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30664,
   "isGpuEnabled": false,
   "isInternetEnabled": false,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 7.689231,
   "end_time": "2024-03-01T09:57:13.088482",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2024-03-01T09:57:05.399251",
   "version": "2.5.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
