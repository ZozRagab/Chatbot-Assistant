"""
One-time script: since the Customer table now requires hashed_password,
this drops and recreates all tables, then re-seeds customers WITH passwords.
Run this once after pulling the updated models.py.
"""
from models import Base, engine, SessionLocal, Customer
from utils import hash_password

# Recreate tables (schema changed - hashed_password column added)
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
print("Tables recreated.")

db = SessionLocal()

# Same 5 customers as before, now with a demo password: "password123" for all
test_customers = [
    (1, "Sarah Ahmed", "sarah.ahmed@example.com"),
    (2, "Mark Johnson", "mark.johnson@example.com"),
    (3, "Lina Farouk", "lina.farouk@example.com"),
    (4, "David Chen", "david.chen@example.com"),
    (5, "Omar Hassan", "omar.hassan@example.com"),
]

for cid, name, email in test_customers:
    customer = Customer(
        id=cid,
        name=name,
        email=email,
        hashed_password=hash_password("password123")
    )
    db.add(customer)

db.commit()
db.close()
print("Customers re-seeded with hashed passwords (all use 'password123' for this demo).")
print("IMPORTANT: you still need to re-run the rest of seed_data.sql")
print("(products, inventory, orders, order_items) since tables were dropped.")