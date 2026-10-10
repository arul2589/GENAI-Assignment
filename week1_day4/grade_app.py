import streamlit as st


st.set_page_config(
    page_title="Grade Manager",
    page_icon="🎓",
    layout="wide",
)

# Keep the student list when Streamlit reruns the script after a user action.
if "students" not in st.session_state:
    st.session_state.students = []


# Same grading scale used in the Day 2 program.
def calculate_grade(mark: int) -> str:
    if mark >= 90:
        return "A"
    elif mark >= 80:
        return "B"
    elif mark >= 70:
        return "C"
    elif mark >= 60:
        return "D"
    return "E"


# Page styling
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 8% 0%, #e6efff 0, transparent 32%),
                    linear-gradient(145deg, #f8faff 0%, #f1f4ff 55%, #f8f4ff 100%);
    }
    .block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero {
        padding: 1.8rem 2rem; border-radius: 24px; color: white;
        background: linear-gradient(115deg, #2447a8, #7354dc);
        box-shadow: 0 15px 38px rgba(55, 67, 160, .20); margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2.4rem; }
    .hero p { margin: .45rem 0 0; opacity: .92; font-size: 1.05rem; }
    div[data-testid="stForm"] {
        padding: 1.3rem 1.4rem; border-radius: 20px; background: rgba(255,255,255,.9);
        border: 1px solid #e3e9fa; box-shadow: 0 10px 28px rgba(40, 57, 110, .07);
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        border: 0; border-radius: 12px; color: white; font-weight: 700;
        background: linear-gradient(100deg, #3457c5, #7654e8); min-height: 2.8rem;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        color: white; border: 0; filter: brightness(1.08);
    }
    div[data-testid="stMetric"] {
        background: white; padding: 1rem 1.2rem; border: 1px solid #e3e9fa;
        border-radius: 17px; box-shadow: 0 8px 22px rgba(40, 57, 110, .06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🎓 Grade Manager</h1>
      <p>Add students, review their grades, and see how the class is doing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("➕ Add a student")
with st.form("add_student_form", clear_on_submit=True):
    name = st.text_input("Student name", placeholder="For example, Jai")
    mark_text = st.text_input("Mark (0–100)", placeholder="Enter a whole number")
    submitted = st.form_submit_button("Add student", use_container_width=True)

if submitted:
    clean_name = name.strip()
    if not clean_name:
        st.error("Please enter the student's name.")
    elif not mark_text.strip():
        st.error("Please enter a mark between 0 and 100.")
    else:
        try:
            mark = int(mark_text.strip())
        except ValueError:
            st.error("Mark must be a whole number, such as 75.")
        else:
            if not 0 <= mark <= 100:
                st.error("Mark must be between 0 and 100.")
            else:
                st.session_state.students.append(
                    {"Name": clean_name, "Mark": mark, "Grade": calculate_grade(mark)}
                )
                st.success(f"Added {clean_name} with grade {calculate_grade(mark)}.")

students = st.session_state.students
st.divider()
st.subheader("📊 Class overview")

if students:
    marks = [student["Mark"] for student in students]
    average = sum(marks) / len(marks)
    first_col, second_col, third_col = st.columns(3)
    first_col.metric("Class average", f"{average:.1f}%", help="Average of all student marks")
    second_col.metric("Highest mark", f"{max(marks)}%")
    third_col.metric("Lowest mark", f"{min(marks)}%")

    st.subheader("👩‍🎓 Student results")
    st.dataframe(students, use_container_width=True, hide_index=True)
else:
    first_col, second_col, third_col = st.columns(3)
    first_col.metric("Class average", "—")
    second_col.metric("Highest mark", "—")
    third_col.metric("Lowest mark", "—")
    st.info("No students yet. Add the first student using the form above.")

with st.expander("View the grade scale"):
    st.markdown(
        """
        | Mark | Grade |
        |:--|:--:|
        | 90–100 | A |
        | 80–89 | B |
        | 70–79 | C |
        | 60–69 | D |
        | 0–59 | E |
        """
    )
