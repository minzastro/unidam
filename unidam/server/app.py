# app.py
from flask import Flask, request, render_template_string, jsonify, session, send_from_directory
import io
from astropy.io.votable import parse_single_table
from astropy.io.votable.tree import Param
from astropy.utils.xml.writer import XMLWriter
import pandas as pd
import os
import uuid

app = Flask(__name__)
app.secret_key = 'your_secret_key'

def runProcessing(votable):
    # Get the current time
    from datetime import datetime
    current_time = datetime.now().isoformat()
    # Update the TIME metadata item
    votable.params.append(Param(votable, 'TIME', 'time', value=current_time))
    return votable


@app.route('/')
def index():
    return render_template_string('''
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <title>Upload VOTable</title>
          </head>
          <body>
            <div class="container">
              <h1>Upload VOTable</h1>
              <form method="post" action="/upload" enctype="multipart/form-data">
                <input type="file" name="votable_file" accept=".xml,.vot">
                <button type="submit">Upload and Process</button>
              </form>
            </div>
          </body>
        </html>
    ''')


@app.route('/upload', methods=['POST'])
def upload():
    if 'votable_file' not in request.files:
        return "No file part", 400
    file = request.files['votable_file']

    if file.filename == '':
        return "No selected file", 400

    if file and file.filename.endswith(('.xml', '.vot', '.votable')):
        # Read the VOTable from the uploaded file
        votable = parse_single_table(file)

        # Process the VOTable
        processed_votable = runProcessing(votable)

        # Convert the updated VOTable to a pandas DataFrame for HTML rendering
        df = processed_votable.to_table().to_pandas()
        session['dataframe'] = df.to_json()
        html_table = df.to_html(classes='table table-striped', index=False)
        html_table = html_table.replace('<tr>', '<tr onclick="openRowData(this)">')

        # Generate a unique filename using UUID and save the file to the cache folder
        filename = str(uuid.uuid4()) + '.vot'
        filepath = os.path.join('.cache', filename)

        with io.StringIO() as s:
            x = XMLWriter(s)
            processed_votable.to_xml(x, version_1_2_or_later=True)
            s.seek(0)
            with open(filepath, 'w') as f:
                f.write(s.read())
        # Create a Flask response with an HTML page containing the table and download button
        html = f'''
            <!doctype html>
            <html lang="en">
              <head>
                <!-- Bootstrap CSS for styling -->
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <!-- Include JavaScript code for handling row click events -->
                <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
                <script>
                    function openRowData(row) {{
                        const index = $(row).index();
                        fetch(`/row_data/${{index}}`)
                            .then(response => response.json())
                            .then(data => {{
                                // Open a new page and display the selected row data
                                const newWindow = window.open('', '_blank');
                                newWindow.document.write('<pre>' + JSON.stringify(data, null, 2) + '</pre>');
                            }});
                    }}
                </script>
              </head>
              <body>
                <div class="container mt-4">
                  <h1>Processed VOTable</h1>
                  <!-- Download button -->
                  <a href="/cache/{filename}" download="processed.vot" class="btn btn-primary mb-3">Download</a>
                  {html_table}
                </div>
              </body>
            </html>
        '''
        return html, 200

@app.route('/row_data/<int:index>', methods=['GET'])
def get_row_data(index):
    # Convert the row at the given index to JSON format and return it
    df = pd.read_json(session['dataframe'])
    return jsonify(df.iloc[index].to_dict())

@app.route('/cache/<path:filename>', methods=['GET'])
def cached_file(filename):
    return send_from_directory('.cache', filename)

if __name__ == '__main__':
    app.run(debug=True)