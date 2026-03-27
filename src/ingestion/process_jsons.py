import os, json
from langchain_core.documents import Document
from config.settings import *

def process_courses_json(json_path):
    documents = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for course in data.get("dersler", []):
        course_name = course.get("ad", "Bilinmeyen Ders")
        semester = course.get("donem", "")
        year = course.get("yil", "")
        course_type = course.get("tip", "")
        easiness_score = course.get("kolaylik_puani", 0)
        necessity_score = course.get("gereklilik_puani", 0)
        
        instructors_list = [h.get("ad") for h in course.get("dersi_veren_hocalar", []) if h.get("ad")]
        instructors_str = ", ".join(instructors_list) if instructors_list else "Belirtilmemiş"
        
        # Base metadata template 
        base_metadata = {
            "source_type": "opinion", # To distinguish from .pdfs
            "course_name": course_name,
            "semester": semester,
            "year": year
        }
        
        if instructors_list:
            base_metadata["instructors"] = instructors_list
        else:
            base_metadata["instructors"] = ["Unknown"] 

        # 1. General information about the course
        general_info_text = (
            f"Ders: {course_name} ({year}. Yıl, {semester} Dönemi, {course_type}). "
            f"Dersi veren hocalar: {instructors_str}. "
            f"Öğrencilerin oylarına göre kolaylık puanı {easiness_score}/100, gereklilik puanı {necessity_score}/100."
        )
        doc_general = Document(
            page_content=general_info_text,
            metadata={**base_metadata, "doc_type": "course_stats"}
        )
        documents.append(doc_general)

        # 2. Student comments
        for opinion in course.get("ogrenci_gorusleri", []):
            comment_text = opinion.get("yorum", "")
            if comment_text:
                enriched_comment = f"{course_name} dersi hakkında öğrenci yorumu/tecrübesi: {comment_text}"
                
                doc_comment = Document(
                    page_content=enriched_comment,
                    metadata={**base_metadata, "doc_type": "student_comment"}
                )
                documents.append(doc_comment)

        # 3. Student advice and suggestions about the course
        for advice_obj in course.get("derse_dair_oneriler", []):
            for advice in advice_obj.get("oneriler", []):
                enriched_advice = f"{course_name} dersini geçmek ve başarılı olmak için öğrenci tavsiyesi: {advice}"
                
                doc_advice = Document(
                    page_content=enriched_advice,
                    metadata={**base_metadata, "doc_type": "student_advice"}
                )
                documents.append(doc_advice)

        # 4. Resources about the course (if available)
        resources = course.get("faydali_olabilecek_kaynaklar", [])
        if resources:
            resources_str = "\n".join(resources)
            resource_text = f"{course_name} dersi için çıkmış sorular, notlar ve faydalı proje kaynakları şunlardır:\n{resources_str}"
            
            doc_resource = Document(
                page_content=resource_text,
                metadata={**base_metadata, "doc_type": "course_resources"}
            )
            documents.append(doc_resource)

    print(f"[INFO] Total of {len(documents)} enriched course chunks created.")
    return documents


def process_semesters_json(json_path):
    documents = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for semester_data in data.get("donemler", []):
        semester_name = semester_data.get("donem_adi", "Bilinmeyen Dönem")
        year = semester_data.get("yil", 0)
        semester = semester_data.get("donem", "")
        
        # Base metadata template
        base_metadata = {
            "source_type": "opinion", 
            "semester_name": semester_name,
            "year": year,
            "semester": semester,
            "doc_type": "semester_advice" 
        }

        # General information about the semester
        advices = semester_data.get("genel_tavsiyeler", [])
        
        if not advices:
            continue
            
        for advice in advices:
            enriched_advice = f"{semester_name} dönemi hakkında genel öğrenci tavsiyesi: {advice}"
            
            doc_advice = Document(
                page_content=enriched_advice,
                metadata=base_metadata
            )
            documents.append(doc_advice)

    print(f"[INFO] Total of {len(documents)} semester advice chunks created.")
    return documents


def process_instructors_json(json_path):
    documents = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for instructor in data.get("hocalar", []):
        instructor_name = instructor.get("ad", "")
        
        if not instructor_name:
            continue
            
        office = instructor.get("ofis", "Belirtilmemiş")
        link = instructor.get("link", "Belirtilmemiş")
        is_active = instructor.get("hoca_aktif_gorevde_mi", True)
        
        expl_score = instructor.get("anlatim_puani", 0)
        ease_score = instructor.get("kolaylik_puani", 0)
        teach_score = instructor.get("ogretme_puani", 0)
        fun_score = instructor.get("eglence_puani", 0)
        
        courses_list = instructor.get("dersler", [])
        courses_str = ", ".join(courses_list) if courses_list else "Belirtilmemiş"
        
        # Base metadata template
        base_metadata = {
            "source_type": "opinion", 
            "instructor_name": instructor_name,
            "is_active": is_active
        }

        # General information about the instructor
        status_text = "Aktif görevde." if is_active else "Şu an aktif görevde değil."
        
        general_info_text = (
            f"Akademisyen: {instructor_name}. Durum: {status_text} "
            f"Ofis: {office}. Araştırma Linki: {link}. "
            f"Verdiği dersler: {courses_str}. "
            f"Öğrenci oylarına göre istatistikleri (100 üzerinden) -> "
            f"Anlatım: {expl_score}, Kolaylık: {ease_score}, Öğretme: {teach_score}, Eğlence: {fun_score}."
        )
        doc_general = Document(
            page_content=general_info_text,
            metadata={**base_metadata, "doc_type": "instructor_stats"}
        )
        documents.append(doc_general)

        # Student comments about the instructor
        for opinion in instructor.get("ogrenci_gorusleri", []):
            comment_text = opinion.get("yorum", "")
            if comment_text:
                enriched_comment = f"{instructor_name} hakkında öğrenci yorumu/tecrübesi: {comment_text}"
                
                doc_comment = Document(
                    page_content=enriched_comment,
                    metadata={**base_metadata, "doc_type": "instructor_comment"}
                )
                documents.append(doc_comment)

    print(f"[INFO] Total of {len(documents)} enriched instructor chunks created.")
    return documents


if __name__ == "__main__":
    # Test the JSON processing functions
    comments_path = os.path.join(DATA_PATH, "student_comments")
    
    courses_docs = process_courses_json(os.path.join(comments_path, "dersler.json"))
    semesters_docs = process_semesters_json(os.path.join(comments_path, "donemler.json"))
    instructors_docs = process_instructors_json(os.path.join(comments_path, "hocalar.json"))

    print(f"Courses documents: {len(courses_docs)}")
    print(f"Semesters documents: {len(semesters_docs)}")
    print(f"Instructors documents: {len(instructors_docs)}")
    
    for doc in courses_docs[:2]:  # Print first 2 course docs as sample
        print(f"\n[COURSE DOC] {doc.metadata['course_name']} - {doc.metadata['semester']} {doc.metadata['year']}\n{doc.page_content}\n")
        
    for doc in semesters_docs[:2]:  # Print first 2 semester docs as sample
        print(f"\n[SEMESTER DOC] {doc.metadata['semester_name']} - {doc.metadata['year']}\n{doc.page_content}\n")
        
    for doc in instructors_docs[:2]:  # Print first 2 instructor docs as sample
        print(f"\n[INSTRUCTOR DOC] {doc.metadata['instructor_name']} (Aktif: {doc.metadata['is_active']})\n{doc.page_content}\n")