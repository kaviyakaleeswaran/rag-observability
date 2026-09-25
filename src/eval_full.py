import json
import sys

# Allow importing pipeline.py from the src folder
sys.path.insert(0, "src")

from pipeline import run_pipeline


EVALUATION_FILE = "data/evaluation.json"
PASS_THRESHOLD = 70.0


def load_evaluation():
    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def get_source_from_chunk_id(chunk_id):
    """
    Extract the document source from a chunk ID.

    Examples:
        resnet_38      -> resnet
        faster_rcnn_0  -> faster_rcnn
        unet_15        -> unet
        yolo_42        -> yolo
        vit_7          -> vit
    """

    parts = chunk_id.rsplit("_", 1)

    if len(parts) != 2:
        return ""

    return parts[0]


def evaluate_question(test_case):
    question = test_case["question"]
    expected_source = test_case["expected_source"]

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)

    result = run_pipeline(question)

    retrieved_chunks = result["retrieved_chunks"]

    retrieved_ids = [
        chunk[0]
        for chunk in retrieved_chunks
    ]

    retrieved_sources = [
        get_source_from_chunk_id(chunk_id)
        for chunk_id in retrieved_ids
    ]

    # Retrieval passes if the expected document appears
    # in the top retrieved chunks.
    retrieval_pass = (
        expected_source in retrieved_sources
    )

    # Citation passes if:
    # 1. At least one citation is present
    # 2. No citation points outside the retrieved chunks
    citation_pass = (
        len(result["valid_citations"]) > 0
        and len(result["invalid_citations"]) == 0
    )

    question_pass = (
        retrieval_pass
        and citation_pass
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nRETRIEVED CHUNKS:")
    print(retrieved_ids)

    print("\nRETRIEVED SOURCES:")
    print(retrieved_sources)

    print("\nEXPECTED SOURCE:")
    print(expected_source)

    print("\nVALID CITATIONS:")
    print(result["valid_citations"])

    print("\nINVALID CITATIONS:")
    print(result["invalid_citations"])

    print("\nRETRIEVAL:", "PASS" if retrieval_pass else "FAIL")
    print("CITATION:", "PASS" if citation_pass else "FAIL")
    print("QUESTION:", "PASS" if question_pass else "FAIL")

    return question_pass


def run_evaluation():
    evaluation_data = load_evaluation()

    total_questions = len(evaluation_data)
    passed_questions = 0

    print("=" * 70)
    print("FULL RAG PIPELINE EVALUATION")
    print("=" * 70)

    for test_case in evaluation_data:

        try:
            passed = evaluate_question(test_case)

            if passed:
                passed_questions += 1

        except Exception as error:
            print("\nERROR WHILE EVALUATING QUESTION:")
            print(error)

    failed_questions = (
        total_questions - passed_questions
    )

    if total_questions > 0:
        pass_rate = (
            passed_questions / total_questions
        ) * 100
    else:
        pass_rate = 0.0

    print("\n\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total questions: {total_questions}")
    print(f"Passed questions: {passed_questions}")
    print(f"Failed questions: {failed_questions}")
    print(f"Pass rate: {pass_rate:.1f}%")
    print(f"Required pass rate: {PASS_THRESHOLD:.0f}%")

    if pass_rate >= PASS_THRESHOLD:

        print("\nEVALUATION PASSED.")
        print(
            "The RAG system meets the required evaluation threshold."
        )

        return True

    else:

        print("\nEVALUATION FAILED.")
        print(
            "The RAG system does not meet the required evaluation threshold."
        )

        return False


if __name__ == "__main__":

    success = run_evaluation()

    if not success:
        sys.exit(1)