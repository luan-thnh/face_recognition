from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

# Cấu hình kết nối cơ sở dữ liệu SQLite
DATABASE_URL = "sqlite:///./checkins.db"

Base = declarative_base()

# Tạo bảng CheckIn


class CheckIn(Base):
    __tablename__ = "checkins"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey('students.id'), nullable=False)
    student_name = Column(String(255), nullable=False)
    checkin_time = Column(
        DateTime, default=datetime.datetime.utcnow, nullable=False)
    status = Column(String(50), default="checked in")

    # Quan hệ với bảng students (nếu cần lấy thông tin sinh viên)
    student = relationship("Student", back_populates="checkins")

# Tạo bảng Student


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)

    # Quan hệ với bảng checkins
    checkins = relationship("CheckIn", back_populates="student")


# Khởi tạo cơ sở dữ liệu và phiên làm việc
engine = create_engine(DATABASE_URL, connect_args={
                       "check_same_thread": False})  # Đặc biệt với SQLite
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Tạo bảng trong cơ sở dữ liệu


def init_db():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()  # Gọi hàm để tạo bảng
    print("Cơ sở dữ liệu đã được tạo thành công.")
