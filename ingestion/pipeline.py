"""
Complete ingestion pipeline:
1. Extract text from PDFs
2. Extract document identity/metadata
3. Group documents into instruments
4. Build collection with chunking
5. Vectorize and push to Qdrant
"""
import sys
import logging
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.text_extractor import run_batch_extraction
from ingestion.doc_identity import run_identity_extraction
from ingestion.instrument_grouper import run_instrument_grouper
from ingestion.collection_builder import build_collection_from_instruments
from ingestion.instrument_vectorizer import run_vectorization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_full_pipeline():
    """Run the complete ingestion pipeline."""
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("🚀 RAG INGESTION PIPELINE - STARTING")
    logger.info("=" * 70)
    logger.info(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    steps = [
        ("1. Text Extraction", run_batch_extraction),
        ("2. Document Identity Extraction", run_identity_extraction),
        ("3. Instrument Grouping", run_instrument_grouper),
        ("4. Collection Building (Chunking)", lambda: build_collection_from_instruments()),
        ("5. Vectorization & Qdrant Upload", run_vectorization),
    ]
    
    step_times = []
    
    for step_num, (step_name, step_func) in enumerate(steps, 1):
        step_start = time.time()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"📋 STEP {step_num}/{len(steps)}: {step_name}")
        logger.info("=" * 70)
        logger.info(f"Step started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            step_func()
            step_duration = time.time() - step_start
            step_times.append((step_name, step_duration))
            
            logger.info("")
            logger.info(f"✅ Step {step_num} completed successfully")
            logger.info(f"⏱️  Step duration: {step_duration:.2f} seconds ({step_duration/60:.2f} minutes)")
            
        except KeyboardInterrupt:
            logger.warning("")
            logger.warning("⚠️  Pipeline interrupted by user")
            logger.warning(f"Interrupted at: {step_name}")
            logger.warning("You can resume from this step later.")
            sys.exit(1)
        except Exception as e:
            logger.error("")
            logger.error(f"❌ Error in step {step_num}: {step_name}")
            logger.error(f"Error message: {str(e)}")
            logger.error("")
            import traceback
            logger.error("Traceback:")
            logger.error(traceback.format_exc())
            sys.exit(1)
    
    total_duration = time.time() - start_time
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    logger.info("")
    logger.info("Step breakdown:")
    for step_name, step_duration in step_times:
        percentage = (step_duration / total_duration) * 100
        logger.info(f"  • {step_name}: {step_duration:.2f}s ({percentage:.1f}%)")
    logger.info("=" * 70)

def run_from_collection():
    """Run only collection building and vectorization (assumes steps 1-3 are done)."""
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("🔄 QUICK UPDATE: Collection & Vectorization")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        step_start = time.time()
        logger.info("")
        logger.info("📋 Step 4: Collection Building (Chunking)")
        logger.info("-" * 70)
        build_collection_from_instruments()
        step_duration = time.time() - step_start
        logger.info(f"✅ Collection building completed in {step_duration:.2f}s")
        
        step_start = time.time()
        logger.info("")
        logger.info("📋 Step 5: Vectorization & Qdrant Upload")
        logger.info("-" * 70)
        run_vectorization()
        step_duration = time.time() - step_start
        logger.info(f"✅ Vectorization completed in {step_duration:.2f}s")
        
        total_duration = time.time() - start_time
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ QUICK UPDATE COMPLETE!")
        logger.info(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error("")
        logger.error(f"❌ Error during quick update: {str(e)}")
        logger.error("")
        import traceback
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_from_collection()
    else:
        run_full_pipeline()
