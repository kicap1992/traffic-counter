from quart import Quart, request, jsonify
import aiomysql
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables from .env file
load_dotenv()

app = Quart(__name__)

# MySQL connection details from environment variables
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_DB = os.getenv('MYSQL_DB')

async def get_db_connection():
    return await aiomysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DB,
        loop=asyncio.get_running_loop()
    )

@app.route('/check-connection', methods=['GET'])
async def check_connection():
    try:
        conn = await get_db_connection()
        conn.close()
        return jsonify({'message': 'Database connection successful!'}), 200
    except Exception as e:
        return jsonify({'message': 'Database connection failed!', 'error': str(e)}), 500

@app.route('/insert', methods=['POST'])
async def insert_data():
    try:
        # Get form data from the POST request
        form = await request.form
        nama = form.get('nama')
        waktu = form.get('waktu')
        kenderaan_kiri = form.get('kenderaan_kiri')
        kenderaan_kanan = form.get('kenderaan_kanan')

        print(nama, waktu, kenderaan_kiri, kenderaan_kanan)

        conn = await get_db_connection()
        async with conn.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO tb_data (nama, waktu, kenderaan_kiri, kenderaan_kanan) VALUES (%s, %s, %s, %s)", 
                (nama, waktu, kenderaan_kiri, kenderaan_kanan)
            )
            await conn.commit()
        conn.close()
        return jsonify({'message': 'Data inserted successfully!'}), 201
    except Exception as e:
        return jsonify({'message': 'Failed to insert data!', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run()



