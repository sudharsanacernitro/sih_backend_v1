import mysql.connector
import json

# Step 1: Establish connection to MySQL
def get_mysql_connection():
    return mysql.connector.connect(
        host="0.0.0.0",  # Replace with your MySQL host
        user="root",  # Replace with your MySQL username
        password="Tatakae-nitro",  # Replace with your MySQL password
        database="Plant_disease_pred"  # Replace with your database name
    )

def add_post_det(data):
    conn = get_mysql_connection()
    cursor = conn.cursor()

    # Step 2: Prepare the SQL query to insert the data into the post_det table
    insert_query = """
    INSERT INTO post_det (district, type, plant_name, coordinates_lat, coordinates_long, img, disease_name, convo)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    district = data.get('district', 'Erode')
    plant_type = data.get('type', 'Fruit')
    plant_name = data.get('plant-name', 'Apple')
    coordinates = data.get('coordinates', [0.0, 0.0])
    img = data.get('img', '')
    disease_name = data.get('Disease-name', '')
    convo = json.dumps(data.get('convo', []))  # Convert convo list to JSON format

    # Step 3: Execute the query and insert the data
    cursor.execute(insert_query, (district, plant_type, plant_name, coordinates[0], coordinates[1], img, disease_name, convo))
    conn.commit()

    # Step 4: Retrieve the last inserted ID
    insertion_id = cursor.lastrowid

    cursor.close()
    conn.close()

    return str(insertion_id)

def update_document(document_id, new_key, new_value):
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()

        # Fetch the document by its ID
        cursor.execute("SELECT convo FROM post_det WHERE id = %s", (document_id,))
        document = cursor.fetchone()

        if document:
            print("Original Document:", document)

            # Update the convo field with the new key-value pair
            convo = json.loads(document[0]) if document[0] else []
            convo.append({new_key: new_value})  # Append the new key-value pair to the convo list

            update_query = "UPDATE post_det SET convo = %s WHERE id = %s"
            cursor.execute(update_query, (json.dumps(convo), document_id))
            conn.commit()

            # Check if the document was successfully updated
            if cursor.rowcount > 0:
                print(f"Document updated with {new_key}: {new_value}")
            else:
                print("No changes made to the document.")
        else:
            print("Document not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cursor.close()
        conn.close()

def find_post():
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()

        # Fetch all documents where district = 'Erode'
        cursor.execute("SELECT * FROM post_det WHERE district = %s", ('Erode',))
        documents = cursor.fetchall()

        cursor.close()
        conn.close()

        return documents
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    # Insert example data
    id = add_post_det({
        'district': "Erode",
        'type': 'Fruit',
        'plant-name': 'Apple',
        'coordinates': [11.2476736, 77.6967361],
        'img': 'img_20240907_154200_675204.jpg',
        'Disease-name': 'apple_scab',
        'convo': [{'sender': 'user', 'Message': 'what is the cause of this disease'}]
    })
    print("Inserted ID:", id)

    # Update example data
    update_document(id, 'remedies', 'Use fungicides in spring when new leaves appear.')
