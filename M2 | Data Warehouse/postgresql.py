import os
import django
from datetime import date

# Configure Django settings for standalone script
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')  # Replace 'your_project' with your Django project name
django.setup()

# Import the FinancialData model
from your_app.models import FinancialData  # Replace 'your_app' with your Django app name

# Function to create a new financial data record
def create_financial_data():
    record = FinancialData(
        date=date(2024, 11, 25),
        asset_name='Bitcoin',
        asset_type='Cryptocurrency',
        opening_price=50000.00,
        closing_price=51000.00,
        volume=1500000
    )
    record.save()
    print(f"Record created with ID: {record.id}")

# Function to retrieve all records
def retrieve_all_data():
    data = FinancialData.objects.all()
    for record in data:
        print(f"{record.date} - {record.asset_name} - {record.closing_price}")

# Function to update a specific record
def update_record(record_id, new_price):
    try:
        record = FinancialData.objects.get(id=record_id)
        record.closing_price = new_price
        record.save()
        print(f"Record {record_id} updated to new price: {new_price}")
    except FinancialData.DoesNotExist:
        print(f"Record with ID {record_id} does not exist.")

# Function to delete a specific record
def delete_record(record_id):
    try:
        record = FinancialData.objects.get(id=record_id)
        record.delete()
        print(f"Record {record_id} deleted.")
    except FinancialData.DoesNotExist:
        print(f"Record with ID {record_id} does not exist.")

# Main function to demonstrate functionality
if __name__ == '__main__':
    print("=== Financial Data Manager ===")
    print("1. Create a new record")
    print("2. Retrieve all records")
    print("3. Update a record")
    print("4. Delete a record")

    choice = input("Choose an option: ")

    if choice == '1':
        create_financial_data()
    elif choice == '2':
        retrieve_all_data()
    elif choice == '3':
        record_id = int(input("Enter the record ID to update: "))
        new_price = float(input("Enter the new closing price: "))
        update_record(record_id, new_price)
    elif choice == '4':
        record_id = int(input("Enter the record ID to delete: "))
        delete_record(record_id)
    else:
        print("Invalid choice. Exiting.")




        