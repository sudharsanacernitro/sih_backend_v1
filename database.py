from pymongo import MongoClient
from bson.objectid import ObjectId

# Step 1: Establish connection to MongoDB
client = MongoClient("mongodb+srv://smartindiahackathon24:soorya@cluster0.aktmx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Replace with your MongoDB URI
db = client["Plant_disease_pred"]  # Replace with your database name

def add_post_det(data):
    collection = db["post_det"]  # Replace with your collection name


    result = collection.insert_one(data)

    # Step 3: Retrieve the insertion ID
    insertion_id = result.inserted_id

    return str(insertion_id)
    print(f"Inserted document ID: {insertion_id}")

def update_document(document_id, new_key, new_value):
    try:
        collection = db["post_det"]
        # Fetch the document by its ObjectId
        document = collection.find_one({"_id": ObjectId(document_id)})
        if document:
            print("Original Document:", document)

            # Update the document with a new key-value pair
            update_result = collection.update_one(
                {"_id": ObjectId(document_id)},  # Find by ObjectId
                {"$set": {new_key: new_value}}   # Add the new key-value pair
            )

            # Check if the document was successfully updated
            if update_result.modified_count > 0:
                print(f"Document updated with {new_key}: {new_value}")
            else:
                print("No changes made to the document.")
        else:
            print("Document not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def find_post():
    try:
        collection=db["correct_predictions"]

        documents = collection.find({"district": "Erode"})

        return list(documents)
    except Exception as e:
        return []
        print(f"An error occurred: {e}")

if __name__=="__main__":
    id=add_post_det({'data':"hai"})
    print(id)