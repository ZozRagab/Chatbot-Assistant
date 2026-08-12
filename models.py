from sqlalchemy import (
    Column, Integer, String, Boolean, Float, Text, DateTime, Date,
    ForeignKey, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('DATABASE_USERNAME')}:{os.getenv('DATABASE_PASSWORD')}"
    f"@{os.getenv('DATABASE_HOSTNAME')}:{os.getenv('DATABASE_PORT')}/{os.getenv('DATABASE_NAME')}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "User"

    Id = Column(Integer, primary_key=True)
    FirstName = Column(String, nullable=False)
    LastName = Column(String, nullable=False)
    Email = Column(String, nullable=False, unique=True)
    PhoneNumber = Column(String)
    HashedPassword = Column(String, nullable=False)
    RefreshToken = Column(String, nullable=True)
    Role = Column(String, nullable=False, default="customer")
    CreatedAt = Column(DateTime, default=datetime.utcnow)
    UpdatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserAddress(Base):
    __tablename__ = "UserAddress"

    Id = Column(Integer, primary_key=True)
    UserId = Column(Integer, ForeignKey("User.Id"), nullable=False)
    Address = Column(String, nullable=False)


class Category(Base):
    __tablename__ = "Category"

    Id = Column(Integer, primary_key=True)
    ParentId = Column(Integer, ForeignKey("Category.Id"), nullable=True)  # self-referencing
    Name = Column(String, nullable=False)
    CreatedAt = Column(DateTime, default=datetime.utcnow)


class Product(Base):
    __tablename__ = "Product"

    Id = Column(Integer, primary_key=True)
    CategoryId = Column(Integer, ForeignKey("Category.Id"), nullable=False)
    Slug = Column(String, nullable=False, unique=True)
    Name = Column(String, nullable=False)
    Description = Column(Text)
    Brand = Column(String)
    Price = Column(Float, nullable=False)
    SalePrice = Column(Float, nullable=True)
    DiscountPercentage = Column(Float, nullable=True)
    StockQuantity = Column(Integer, nullable=False, default=0)
    Ingredients = Column(Text, nullable=True)
    isActive = Column(Boolean, default=True)
    ProductImage = Column(String, nullable=True)
    AltText = Column(String, nullable=True)


class Tag(Base):
    __tablename__ = "Tag"

    Id = Column(Integer, primary_key=True)
    Name = Column(String, nullable=False, unique=True)


class ProductTags(Base):
    __tablename__ = "ProductTags"

    Id = Column(Integer, primary_key=True)
    ProductId = Column(Integer, ForeignKey("Product.Id"), nullable=False)
    TagId = Column(Integer, ForeignKey("Tag.Id"), nullable=False)


class Review(Base):
    __tablename__ = "Review"

    ID = Column(Integer, primary_key=True)
    UserId = Column(Integer, ForeignKey("User.Id"), nullable=False)
    ProductId = Column(Integer, ForeignKey("Product.Id"), nullable=False)
    Rating = Column(Integer, nullable=False)
    Comment = Column(Text, nullable=True)
    CreationDate = Column(DateTime, default=datetime.utcnow)


class Cart_Item(Base):
    __tablename__ = "Cart_Item"

    Id = Column(Integer, primary_key=True)
    CartId = Column(Integer, ForeignKey("Cart.Id"), nullable=False)
    ProductId = Column(Integer, ForeignKey("Product.Id"), nullable=False)
    Quantity = Column(Integer, nullable=False, default=1)


class Cart(Base):
    __tablename__ = "Cart"

    Id = Column(Integer, primary_key=True)
    # As drawn in the ERD: Cart also holds a direct FK to a Cart_Item.
    # use_alter=True handles the circular FK dependency at table-creation time.
    CartItemId = Column(
        Integer,
        ForeignKey("Cart_Item.Id", use_alter=True, name="fk_cart_cartitem"),
        nullable=True
    )
    UserId = Column(Integer, ForeignKey("User.Id"), nullable=False)


class Voucher(Base):
    __tablename__ = "Voucher"

    VoucherId = Column(Integer, primary_key=True)
    Code = Column(String, nullable=False, unique=True)
    ExpiryDate = Column(Date, nullable=False)
    IsExpired = Column(Boolean, default=False)
    Amount = Column(Float, nullable=False)


class Order(Base):
    __tablename__ = "Orders"  # renamed ONLY because ORDER is a reserved SQL keyword

    Id = Column(Integer, primary_key=True)
    UserId = Column(Integer, ForeignKey("User.Id"), nullable=False)
    VoucherId = Column(Integer, ForeignKey("Voucher.VoucherId"), nullable=True)
    AddressId = Column(Integer, ForeignKey("UserAddress.Id"), nullable=False)
    IdempotenceKey = Column(String, nullable=True, unique=True)
    TotalAmount = Column(Float, nullable=False)
    Status = Column(String, nullable=False, default="processing")
    PaymentMethod = Column(String, nullable=True)
    CreationDate = Column(DateTime, default=datetime.utcnow)
    DeliveryDate = Column(DateTime, nullable=True)


class Order_Item(Base):
    __tablename__ = "Order_Item"

    Id = Column(Integer, primary_key=True)
    ProductId = Column(Integer, ForeignKey("Product.Id"), nullable=False)
    OrderId = Column(Integer, ForeignKey("Orders.Id"), nullable=False)
    Quantity = Column(Integer, nullable=False)
    UnitPrice = Column(Float, nullable=False)


def get_db():
    """FastAPI dependency - yields a DB session and ensures it's closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Tables created: User, UserAddress, Category, Product, Tag, ProductTags, "
          "Review, Cart, Cart_Item, Voucher, Orders, Order_Item")