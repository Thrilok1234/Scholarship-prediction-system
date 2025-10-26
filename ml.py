import webbrowser

# === STEP 7: Prediction ===
if submitted:
    sample = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "state": state,
        "category": category,
        "family_income_annual_inr": income,
        "disability": disability,
        "institution_type": institution,
        "program": program,
        "branch": branch,
        "year": year,
        "final_year": final_year,
        "tenth_percent": tenth,
        "twelfth_percent": twelfth,
        "ug_cgpa": cgpa,
        "gate_score": gate_score
    }])

    pred = clf.predict(sample)
    mlb = joblib.load("label_binarizer.pkl")
    pred_labels = mlb.inverse_transform(pred)
    scholarships = pred_labels[0]

    scholarship_links = {
        "Post-Matric Scholarship (SC/ST/OBC)": "https://scholarships.gov.in",
        "State Post-Matric Scholarship": "https://scholarships.gov.in",
        "GATE-based MTech Fellowship": "https://aicte-india.org/schemes/students-development-schemes/PG-Scholarship",
        "NTPC Scholarship": "https://www.ntpc.co.in/en/careers/scholarships",
        "AICTE Pragati Scholarship": "https://aicte-india.org/schemes/students-development-schemes/Pragati-Scheme",
        "Central Sector Scholarship": "https://scholarships.gov.in",
        "Siemens Scholarship Program": "https://www.siemens.co.in/en/scholarship",
        "Tata Trusts Scholarship": "https://www.tatatrusts.org/our-work/education/scholarships"
    }

    st.subheader("🎯 Eligible Scholarships:")
    if scholarships:
        for s in scholarships:
            link = scholarship_links.get(s)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{s}**")
            with col2:
                if st.button("Open", key=s):
                    webbrowser.open(link)
    else:
        st.info("No matching scholarships found for the entered details.")
