from student.reader.reader import Reader, ReaderError
from student.models import (
    StudentSearchResults,
    RagDataset,
    StudentSearchResultsAndAnswer,
    AnsweredQuestion,
    MinimalSource)
from pydantic import ValidationError


class EvaluatorError(Exception):
    pass


class Evaluator:
    def __init__(
            self,
            answers_path: str,
            student_path: str,
            ) -> None:
        self.reader = Reader()
        try:
            answers_content, _ = self.reader.validate_read(answers_path)
            stu_content, _ = self.reader.validate_read(student_path)
            answers = RagDataset.model_validate_json(answers_content)
            for answer in answers.rag_questions:
                if not isinstance(answer, AnsweredQuestion):
                    raise EvaluatorError(
                        "Answers must be valid AnsweredQuestion objects")
            try:
                student = StudentSearchResults.model_validate_json(stu_content)
            except ValidationError:
                student = StudentSearchResultsAndAnswer.model_validate_json(
                    stu_content
                )
            self.student = {
                stud_answ.question_id: stud_answ
                for stud_answ in student.search_results
            }
            self.answers = {
                answer.question_id: answer for answer in answers.rag_questions
            }
        except (ReaderError, ValidationError) as e:
            raise EvaluatorError(e)

    def analyze(self, k: int) -> float:
        if not self.student:
            raise EvaluatorError("No student results")
        if not self.answers:
            raise EvaluatorError("No answer results")

        question_score = []
        for question_id, stu_answer in self.student.items():
            answer = self.answers.get(question_id, None)
            if answer is None:
                continue
            if not isinstance(answer, AnsweredQuestion):
                continue
            found = 0
            for ans_source in answer.sources:
                source_found = False
                for stu_source in stu_answer.retrieved_sources[:k]:
                    recall = self.sources_analyzer(ans_source, stu_source)
                    if recall >= 0.05:
                        source_found = True
                        break
                if source_found:
                    found += 1
            question_score.append(found / len(answer.sources))

        if not question_score:
            return 0.0

        return sum(question_score) / len(question_score)

    @staticmethod
    def sources_analyzer(
        src_answer: MinimalSource,
        src_student: MinimalSource
    ) -> float:
        if src_answer.file_path != src_student.file_path:
            return 0.0

        minval = min(
            src_answer.first_character_index,
            src_student.first_character_index
        )
        maxval = max(
            src_answer.last_character_index,
            src_student.last_character_index
        )

        max_range = maxval - minval

        inter_start = max(
            src_answer.first_character_index,
            src_student.first_character_index,
        )
        inter_end = min(
            src_answer.last_character_index,
            src_student.last_character_index,
        )
        intersection = max(0, inter_end - inter_start)
        if max_range <= 0:
            return 0.0
        return intersection / max_range
