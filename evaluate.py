"""
Evaluation script for RAG system performance

This script evaluates the effectiveness of the Victorian Transport RAG system by:
1. Processing a set of predefined FAQ questions
2. Measuring response quality and accuracy
3. Analyzing source document retrieval
4. Calculating performance metrics
5. Generating detailed evaluation reports

Key Metrics Evaluated:
- Response accuracy and completeness
- Source document relevance
- Processing time and system efficiency
- Answer confidence levels
- Category-wise performance analysis
"""
import json
import os
import sys
import time
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from run_rag import VictorianTransportRAG


class RAGEvaluator:
    """
    Evaluate RAG system performance on FAQ questions
    
    This class handles comprehensive evaluation of the RAG system by:
    - Processing predefined test questions from JSON
    - Measuring response quality and timing
    - Analyzing source document retrieval effectiveness
    - Calculating performance metrics across categories
    - Generating detailed evaluation reports
    """
    
    def __init__(self, rag_system: VictorianTransportRAG, faq_file: str = "examples/faq_samples.json"):
        self.rag_system = rag_system
        self.faq_file = faq_file
        self.results = []
    
    def load_test_questions(self) -> List[Dict]:
        """
        Load test questions from JSON file
        
        Returns:
            List[Dict]: List of question dictionaries, each containing:
                - id: Unique question identifier
                - category: Question category
                - question: The actual question text
                - difficulty: Optional difficulty rating
        """
        if not os.path.exists(self.faq_file):
            print(f"FAQ file not found: {self.faq_file}")
            return []
        
        with open(self.faq_file, 'r') as f:
            data = json.load(f)
        
        return data.get('faq_samples', [])
    
    def evaluate_single_question(self, question_data: Dict) -> Dict[str, Any]:
        """
        Evaluate a single question through the RAG system
        
        Args:
            question_data: Dictionary containing question details and metadata
        
        Returns:
            Dict containing evaluation metrics:
            - Basic: response time, answer length, source count
            - Quality: confidence, source diversity
            - Metadata: timestamp, question category, difficulty
        """
        question = question_data['question']
        
        print(f"Evaluating: {question}")
        
        start_time = time.time()
        
        try:
            # Get response from RAG system
            response = self.rag_system.ask_question(question)
            response_time = time.time() - start_time
            
            # Basic metrics
            answer = response['answer']
            sources = response['source_documents']
            
            # Calculate metrics
            metrics = {
                'question_id': question_data['id'],
                'category': question_data['category'],
                'question': question,
                'answer': answer,
                'response_time': response_time,
                'num_sources': len(sources),
                'answer_length': len(answer),
                'has_answer': len(answer.strip()) > 0,
                'sources_retrieved': len(sources) > 0,
                'difficulty': question_data.get('difficulty', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Check for "don't know" responses
            dont_know_phrases = [
                "don't have", "don't know", "not available", "can't find",
                "unable to", "no information", "not specified", "unclear"
            ]
            
            metrics['confident_answer'] = not any(phrase in answer.lower() for phrase in dont_know_phrases)
            
            # Source quality assessment
            if sources:
                source_names = [s['source'] for s in sources]
                metrics['unique_sources'] = len(set(source_names))
                metrics['source_diversity'] = len(set(source_names)) / len(sources)
            else:
                metrics['unique_sources'] = 0
                metrics['source_diversity'] = 0.0
                
        except Exception as e:
            print(f"Error evaluating question: {str(e)}")
            metrics = {
                'question_id': question_data['id'],
                'category': question_data['category'], 
                'question': question,
                'error': str(e),
                'response_time': time.time() - start_time,
                'has_answer': False,
                'timestamp': datetime.now().isoformat()
            }
        
        return metrics
    
    def run_evaluation(self) -> pd.DataFrame:
        """Run evaluation on all test questions"""
        print("Loading test questions...")
        questions = self.load_test_questions()
        
        if not questions:
            print("No test questions found!")
            return pd.DataFrame()
        
        print(f"Running evaluation on {len(questions)} questions...")
        
        self.results = []
        
        for i, question_data in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}]", end=" ")
            result = self.evaluate_single_question(question_data)
            self.results.append(result)
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.5)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        return df
    
    def calculate_summary_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary metrics from evaluation results
        
        Processes evaluation data to generate comprehensive metrics including:
        - Coverage: success rate, answer coverage
        - Quality: confidence rate, source retrieval
        - Performance: response times, answer lengths
        - Analysis: category and difficulty breakdowns
        
        Args:
            df: DataFrame containing raw evaluation results
        
        Returns:
            Dictionary of calculated metrics and statistical summaries
        """
        if df.empty:
            return {}
        
        # Filter out error cases for accurate metrics
        valid_df = df[~df['answer'].isna() & (df['error'].isna() if 'error' in df.columns else True)]
        
        if valid_df.empty:
            return {"error": "No valid responses to analyze"}
        
        summary = {
            # Coverage metrics
            'total_questions': len(df),
            'successful_responses': len(valid_df),
            'success_rate': len(valid_df) / len(df) * 100,
            
            # Answer quality metrics  
            'questions_with_answers': valid_df['has_answer'].sum(),
            'answer_coverage': valid_df['has_answer'].mean() * 100,
            'confident_answers': valid_df['confident_answer'].sum() if 'confident_answer' in valid_df.columns else 0,
            'confidence_rate': valid_df['confident_answer'].mean() * 100 if 'confident_answer' in valid_df.columns else 0,
            
            # Response characteristics
            'avg_response_time': valid_df['response_time'].mean(),
            'avg_answer_length': valid_df['answer_length'].mean() if 'answer_length' in valid_df.columns else 0,
            'avg_sources_per_question': valid_df['num_sources'].mean() if 'num_sources' in valid_df.columns else 0,
            
            # Source quality
            'questions_with_sources': valid_df['sources_retrieved'].sum() if 'sources_retrieved' in valid_df.columns else 0,
            'source_retrieval_rate': valid_df['sources_retrieved'].mean() * 100 if 'sources_retrieved' in valid_df.columns else 0,
            
            # Category breakdown
            'performance_by_category': valid_df.groupby('category')['has_answer'].mean().to_dict() if 'category' in valid_df.columns else {},
            'performance_by_difficulty': valid_df.groupby('difficulty')['has_answer'].mean().to_dict() if 'difficulty' in valid_df.columns else {}
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print evaluation summary"""
        print(f"\n{'='*60}")
        print("RAG SYSTEM EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        if 'error' in summary:
            print(f"Error: {summary['error']}")
            return
        
        print(f"📊 COVERAGE METRICS:")
        print(f"  Total Questions: {summary['total_questions']}")
        print(f"  Successful Responses: {summary['successful_responses']}")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Answer Coverage: {summary['answer_coverage']:.1f}%")
        print(f"  Confidence Rate: {summary['confidence_rate']:.1f}%")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"  Avg Response Time: {summary['avg_response_time']:.2f}s")
        print(f"  Avg Answer Length: {summary['avg_answer_length']:.0f} chars")
        print(f"  Avg Sources Retrieved: {summary['avg_sources_per_question']:.1f}")
        print(f"  Source Retrieval Rate: {summary['source_retrieval_rate']:.1f}%")
        
        if summary['performance_by_category']:
            print(f"\n📂 PERFORMANCE BY CATEGORY:")
            for category, score in summary['performance_by_category'].items():
                print(f"  {category.title()}: {score:.1f}%")
        
        if summary['performance_by_difficulty']:
            print(f"\n🎯 PERFORMANCE BY DIFFICULTY:")
            for difficulty, score in summary['performance_by_difficulty'].items():
                print(f"  {difficulty.title()}: {score:.1f}%")
    
    def save_results(self, df: pd.DataFrame, filename: str = None):
        """
        Save evaluation results to CSV file
        
        Args:
            df: DataFrame containing evaluation results
            filename: Optional custom filename, defaults to timestamp-based name
                     Format: evaluation_results_YYYYMMDD_HHMMSS.csv
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")


def main():
    """
    Main evaluation function
    
    Orchestrates the complete evaluation process:
    1. Initializes the RAG system
    2. Loads and processes required documents
    3. Runs evaluation on test questions
    4. Generates performance metrics
    5. Saves results to file
    6. Displays sample outputs
    """
    print("RAG System Evaluation")
    print("====================")
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = VictorianTransportRAG()
    
    # Load documents
    if not rag.load_and_process_documents():
        print("Failed to load documents!")
        return
    
    # Setup QA chain
    rag.setup_qa_chain()
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag)
    
    # Run evaluation
    results_df = evaluator.run_evaluation()
    
    if results_df.empty:
        print("No evaluation results to analyze!")
        return
    
    # Calculate and print summary
    summary = evaluator.calculate_summary_metrics(results_df)
    evaluator.print_summary(summary)
    
    # Save results
    evaluator.save_results(results_df)
    
    # Show sample results
    print(f"\n📝 SAMPLE RESULTS:")
    print("="*60)
    
    for _, row in results_df.head(3).iterrows():
        print(f"\nQ: {row['question']}")
        if 'error' not in row or pd.isna(row.get('error')):
            print(f"A: {row['answer'][:200]}...")
            print(f"⏱️  {row['response_time']:.2f}s | 📄 {row.get('num_sources', 0)} sources")
        else:
            print(f"❌ Error: {row.get('error', 'Unknown error')}")
        print("-" * 40)


if __name__ == "__main__":
    main()