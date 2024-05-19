import azure.functions as func
import datetime
import json
import logging
from json import JSONEncoder
import cohere
import os

app = func.FunctionApp()

class DateTimeEncoder(JSONEncoder):
    # Override the default method
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()

def compose_response(json_data):
    values = json.loads(json_data)["values"]

    # Prepare the Output before the loop
    results = {}
    results["values"] = []

    for value in values:
        output_record = analyze_document(recordId=value["recordId"], data=value["data"])
        results["values"].append(output_record)

    return json.dumps(results, ensure_ascii=False, cls=DateTimeEncoder)


def generate_embeddings_cohere(text, input_type):
    url = os.environ["COHERE_EMBED_ENDPOINT"]
    api_key = os.environ["COHERE_EMBED_KEY"]

    co_peygo = cohere.Client(base_url=url, api_key=api_key)

    logging.info("input_type is " + input_type)
    response = co_peygo.embed(texts=[text], input_type=input_type)
    return response.embeddings[0]


def analyze_document(recordId, data):
    try:
        text = data["text"]
        input_type = data.get("input_type", "search_query")
        output_record = {}
        output_record_data = {}

        embeddings = generate_embeddings_cohere(text, input_type)

        output_record_data = {"vector": embeddings}
        output_record = {"recordId": recordId, "data": output_record_data}

    except Exception as error:
        output_record = {
            "recordId": recordId,
            "errors": [{"message": "Error: " + str(error)}],
        }

    # logging.info("Output record: " + json.dumps(output_record, ensure_ascii=False, cls=DateTimeEncoder))
    return output_record


@app.route(route="embeddings", auth_level=func.AuthLevel.FUNCTION)
def embeddings(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Invoked Cohere v3 embeddings function")
    try:
        body = json.dumps(req.get_json(), ensure_ascii=False)

        if body:
            logging.info(body)
            result = compose_response(body)
            logging.info("Result to return to custom skill")
            return func.HttpResponse(result, mimetype="application/json")
        else:
            return func.HttpResponse("Invalid body", status_code=400)
    except ValueError:
        return func.HttpResponse("Invalid body", status_code=400)
