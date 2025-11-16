import streamlit as st
import sqlite3
import csv
import urllib.request
import pandas as pd
from typing import List, Dict
import os

# Page configuration
st.set_page_config(
    page_title="Orchid Database Search",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #ff69b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff69b4;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class OrchidSearchDB:
    """SQLite-based orchid database with Full-Text Search (FTS5) optimization"""
    
    def __init__(self, db_path: str = "orchids.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            
    def create_tables(self):
        """Create main table and FTS5 virtual table"""
        # Main orchid data table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS orchids (
                id INTEGER PRIMARY KEY,
                Species_Key INTEGER,
                Scientific_Name TEXT,
                Canonical_Name TEXT,
                Genus TEXT,
                Species_Epithet TEXT,
                Author TEXT,
                Taxonomic_Status TEXT,
                Kingdom TEXT,
                Family TEXT,
                Subfamily TEXT,
                Order_Name TEXT,
                Growth_Habit TEXT,
                Flower_Size_cm TEXT,
                Flower_Color TEXT,
                Petal_Shape TEXT,
                Petal_Count TEXT,
                Lip_Shape TEXT,
                Lip_Color TEXT,
                Column_Structure TEXT,
                Fragrance TEXT,
                Fragrance_Description TEXT,
                Blooming_Season TEXT,
                Bloom_Duration_Weeks TEXT,
                Flowers_Per_Spike TEXT,
                Spike_Length_cm TEXT,
                Light_Requirement_FC TEXT,
                Light_Description TEXT,
                Temperature_Min_C REAL,
                Temperature_Max_C REAL,
                Temperature_Preference TEXT,
                Humidity_Min_Percent INTEGER,
                Humidity_Max_Percent INTEGER,
                Watering_Frequency TEXT,
                Fertilizer_Requirement TEXT,
                Potting_Media TEXT,
                Pseudobulb TEXT,
                Pseudobulb_Shape TEXT,
                Stem_Type TEXT,
                Stem_Length_cm TEXT,
                Leaf_Type TEXT,
                Leaf_Length_cm TEXT,
                Leaf_Color TEXT,
                Leaf_Arrangement TEXT,
                Root_Type TEXT,
                Root_Color TEXT,
                Pollination_Type TEXT,
                Pollination_Mechanism TEXT,
                Seed_Type TEXT,
                Propagation_Method TEXT,
                Native_Habitat TEXT,
                Native_Regions TEXT,
                Elevation_Min_m INTEGER,
                Elevation_Max_m INTEGER,
                Climate_Type TEXT,
                Rainfall_Requirement TEXT,
                Air_Movement TEXT,
                Mycorrhizal_Association TEXT,
                Conservation_Status TEXT,
                Threatened_Level TEXT,
                Horticultural_Difficulty TEXT,
                Horticultural_Notes TEXT,
                Commercial_Importance TEXT,
                Breeding_Potential TEXT,
                Disease_Susceptibility TEXT,
                Pest_Susceptibility TEXT,
                Special_Features TEXT,
                Cultural_Significance TEXT,
                Common_Names TEXT,
                Etymology TEXT
            )
        """)
        
        # FTS5 virtual table for full-text search
        self.cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS orchids_fts USING fts5(
                Scientific_Name,
                Genus,
                Flower_Color,
                Petal_Shape,
                Lip_Color,
                Fragrance_Description,
                Blooming_Season,
                Temperature_Preference,
                Native_Habitat,
                Native_Regions,
                Special_Features,
                Common_Names,
                Horticultural_Notes,
                content=orchids,
                content_rowid=id
            )
        """)
        
        # Create indexes for optimized filtering
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_genus ON orchids(Genus)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_flower_color ON orchids(Flower_Color)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_temp ON orchids(Temperature_Min_C, Temperature_Max_C)")
        
        self.conn.commit()
        
    def load_data_from_url(self, url: str):
        """Load CSV data from URL and populate database"""
        with urllib.request.urlopen(url) as response:
            lines = [line.decode('utf-8') for line in response.readlines()]
        
        reader = csv.DictReader(lines)
        
        # Get the actual column names from CSV
        csv_columns = reader.fieldnames
        
        count = 0
        for row in reader:
            # Build dynamic INSERT query based on CSV columns
            columns = []
            values = []
            
            for col in csv_columns:
                # Map CSV column names to database column names
                db_col = col
                if col == 'Order':
                    db_col = 'Order_Name'  # 'Order' is a SQL keyword, so we renamed it
                
                columns.append(db_col)
                values.append(row[col])
            
            placeholders = ', '.join(['?' for _ in columns])
            columns_str = ', '.join(columns)
            
            self.cursor.execute(f"""
                INSERT INTO orchids ({columns_str})
                VALUES ({placeholders})
            """, values)
            count += 1
            
        self.conn.commit()
        
        # Populate FTS table manually
        print("Building FTS index...")
        self.cursor.execute("""
            INSERT INTO orchids_fts (rowid, Scientific_Name, Genus, Flower_Color, 
                                     Petal_Shape, Lip_Color, Fragrance_Description,
                                     Blooming_Season, Temperature_Preference, 
                                     Native_Habitat, Native_Regions, Special_Features,
                                     Common_Names, Horticultural_Notes)
            SELECT id, Scientific_Name, Genus, Flower_Color, 
                   Petal_Shape, Lip_Color, Fragrance_Description,
                   Blooming_Season, Temperature_Preference, 
                   Native_Habitat, Native_Regions, Special_Features,
                   Common_Names, Horticultural_Notes
            FROM orchids
        """)
        self.conn.commit()
        print(f"FTS index built for {count} records")
        
        return count
        
    def fulltext_search(self, query: str, limit: int = 50) -> List[Dict]:
        """Perform full-text search using FTS5 MATCH syntax"""
        try:
            # Sanitize query - remove problematic characters
            query = query.strip()
            if not query:
                return []
            
            sql = """
                SELECT o.*
                FROM orchids o
                WHERE o.id IN (
                    SELECT rowid FROM orchids_fts 
                    WHERE orchids_fts MATCH ?
                )
                LIMIT ?
            """
            self.cursor.execute(sql, (query, limit))
            return [dict(row) for row in self.cursor.fetchall()]
        except Exception as e:
            # Fallback to LIKE search if FTS fails
            print(f"FTS search failed: {e}, falling back to LIKE search")
            return self.fallback_search(query, limit)
    
    def fallback_search(self, query: str, limit: int = 50) -> List[Dict]:
        """Fallback search using LIKE when FTS fails"""
        sql = """
            SELECT * FROM orchids 
            WHERE Scientific_Name LIKE ? 
               OR Genus LIKE ?
               OR Flower_Color LIKE ?
               OR Common_Names LIKE ?
               OR Native_Regions LIKE ?
               OR Special_Features LIKE ?
            LIMIT ?
        """
        search_term = f"%{query}%"
        self.cursor.execute(sql, (search_term, search_term, search_term, 
                                  search_term, search_term, search_term, limit))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def semantic_search(self, limit: int = 50, **filters) -> List[Dict]:
        """Search with semantic filters"""
        conditions = []
        params = []
        
        if 'genus' in filters and filters['genus']:
            conditions.append("Genus LIKE ?")
            params.append(f"%{filters['genus']}%")
            
        if 'flower_color' in filters and filters['flower_color']:
            conditions.append("Flower_Color LIKE ?")
            params.append(f"%{filters['flower_color']}%")
            
        if 'min_temp' in filters and filters['min_temp'] is not None:
            conditions.append("Temperature_Min_C >= ?")
            params.append(filters['min_temp'])
            
        if 'max_temp' in filters and filters['max_temp'] is not None:
            conditions.append("Temperature_Max_C <= ?")
            params.append(filters['max_temp'])
            
        if 'native_region' in filters and filters['native_region']:
            conditions.append("Native_Regions LIKE ?")
            params.append(f"%{filters['native_region']}%")
            
        if 'fragrance' in filters and filters['fragrance']:
            if filters['fragrance'].lower() == 'fragrant':
                conditions.append("(Fragrance LIKE ? OR Fragrance LIKE ?)")
                params.extend(['%fragrant%', '%Fragrant%'])
            else:
                conditions.append("Fragrance LIKE ?")
                params.append(f"%{filters['fragrance']}%")
        
        if 'difficulty' in filters and filters['difficulty']:
            conditions.append("Horticultural_Difficulty LIKE ?")
            params.append(f"%{filters['difficulty']}%")
            
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"SELECT * FROM orchids WHERE {where_clause} LIMIT ?"
        params.append(limit)
        
        self.cursor.execute(sql, params)
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        if conditions:
            base_query += " AND (" + " AND ".join(conditions) + ")"
        
        base_query += f" LIMIT ?"
        params.append(limit)
        
        self.cursor.execute(base_query, params)
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        # Total count
        self.cursor.execute("SELECT COUNT(*) as count FROM orchids")
        stats['total'] = self.cursor.fetchone()['count']
        
        # Unique genera
        self.cursor.execute("SELECT COUNT(DISTINCT Genus) as count FROM orchids")
        stats['genera'] = self.cursor.fetchone()['count']
        
        # Colors
        self.cursor.execute("SELECT COUNT(DISTINCT Flower_Color) as count FROM orchids")
        stats['colors'] = self.cursor.fetchone()['count']
        
        # Regions
        self.cursor.execute("SELECT COUNT(DISTINCT Native_Regions) as count FROM orchids")
        stats['regions'] = self.cursor.fetchone()['count']
        
        return stats
    
    def get_unique_values(self, column: str) -> List[str]:
        """Get unique values for a column"""
        self.cursor.execute(f"SELECT DISTINCT {column} FROM orchids WHERE {column} IS NOT NULL ORDER BY {column}")
        return [row[0] for row in self.cursor.fetchall()]

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = None
    st.session_state.data_loaded = False

# Initialize database
@st.cache_resource
def init_database():
    db = OrchidSearchDB()
    db.connect()
    db.create_tables()
    
    # Check if data exists
    db.cursor.execute("SELECT COUNT(*) as count FROM orchids")
    count = db.cursor.fetchone()['count']
    
    if count == 0:
        # Automatically load data
        try:
            url = "https://raw.githubusercontent.com/rashadul-se/orchids/refs/heads/main/orchid_complete_dataset_67fields_2025-11-16.csv"
            count = db.load_data_from_url(url)
            return db, True, count
        except Exception as e:
            return db, False, str(e)
    return db, True, count

db, data_loaded, load_info = init_database()
st.session_state.db = db
st.session_state.data_loaded = data_loaded

# Header
st.markdown('<div class="main-header">🌸 Orchid Database Search 🌸</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f33a.png", width=100)
    st.title("Database Management")
    
# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f33a.png", width=100)
    st.title("Database Management")
    
    # Show loading status
    if st.session_state.data_loaded:
        st.success("✅ Database loaded!")
        if isinstance(load_info, int):
            st.info(f"📊 {load_info} records loaded")
        
        # Statistics
        stats = db.get_statistics()
        st.markdown("### 📊 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", stats['total'])
            st.metric("Unique Genera", stats['genera'])
        with col2:
            st.metric("Flower Colors", stats['colors'])
            st.metric("Native Regions", stats['regions'])
        
        # Reset database option
        st.markdown("---")
        if st.button("🔄 Reset Database", help="Clear all data and reload"):
            try:
                db.cursor.execute("DELETE FROM orchids")
                db.cursor.execute("DELETE FROM orchids_fts")
                db.conn.commit()
                st.cache_resource.clear()
                st.success("Database reset! Refreshing...")
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting database: {str(e)}")
    else:
        st.error("❌ Failed to load database")
        if isinstance(load_info, str):
            st.error(f"Error: {load_info}")
        
        if st.button("🔄 Retry Loading", type="primary"):
            st.cache_resource.clear()
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🔍 Search Tips")
    st.markdown("""
    **Full-Text Search:**
    - Simple: `pink`
    - AND: `pink AND fragrant`
    - OR: `white OR yellow`
    - NOT: `pink NOT purple`
    - Phrase: `"moth orchid"`
    - Prefix: `phala*`
    """)

# Main content
if st.session_state.data_loaded:
    # Search tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Full-Text Search", "🎯 Advanced Filter", "🔗 Combined Search", "📊 Browse All"])
    
    # Tab 1: Full-Text Search
    with tab1:
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            fts_query = st.text_input(
                "Enter search query",
                placeholder="e.g., pink AND fragrant, phala*, 'moth orchid'",
                help="Supports AND, OR, NOT operators and prefix matching with *"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if fts_query and search_btn:
            with st.spinner("Searching..."):
                try:
                    results = db.fulltext_search(fts_query, limit=50)
                    
                    if results:
                        st.success(f"Found {len(results)} results")
                        
                        # Display results
                        for i, result in enumerate(results, 1):
                            with st.expander(f"**{i}. {result['Scientific_Name']}** - {result.get('Common_Names', 'N/A')}"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown("**🌺 Flower Details**")
                                    st.write(f"**Color:** {result.get('Flower_Color', 'N/A')}")
                                    st.write(f"**Size:** {result.get('Flower_Size_cm', 'N/A')} cm")
                                    st.write(f"**Shape:** {result.get('Petal_Shape', 'N/A')}")
                                    st.write(f"**Fragrance:** {result.get('Fragrance', 'N/A')}")
                                
                                with col2:
                                    st.markdown("**🌡️ Growing Conditions**")
                                    st.write(f"**Temperature:** {result.get('Temperature_Min_C', 'N/A')}-{result.get('Temperature_Max_C', 'N/A')}°C")
                                    st.write(f"**Humidity:** {result.get('Humidity_Min_Percent', 'N/A')}-{result.get('Humidity_Max_Percent', 'N/A')}%")
                                    st.write(f"**Light:** {result.get('Light_Description', 'N/A')}")
                                    st.write(f"**Difficulty:** {result.get('Horticultural_Difficulty', 'N/A')}")
                                
                                with col3:
                                    st.markdown("**🌍 Origin & Habitat**")
                                    st.write(f"**Region:** {result.get('Native_Regions', 'N/A')}")
                                    st.write(f"**Habitat:** {result.get('Native_Habitat', 'N/A')}")
                                    st.write(f"**Climate:** {result.get('Climate_Type', 'N/A')}")
                                    st.write(f"**Blooming:** {result.get('Blooming_Season', 'N/A')}")
                                
                                if result.get('Special_Features'):
                                    st.info(f"✨ **Special Features:** {result['Special_Features']}")
                    else:
                        st.warning("No results found. Try a different query or simpler search terms.")
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
                    st.info("💡 Try simpler search terms like: pink, fragrant, or Phalaenopsis")
    
    # Tab 2: Advanced Filter
    with tab2:
        st.markdown("### Filter by Specific Criteria")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            genus = st.selectbox("Genus", [""] + db.get_unique_values("Genus"))
            flower_color = st.text_input("Flower Color", placeholder="e.g., Pink, White")
            fragrance = st.selectbox("Fragrance", ["", "Fragrant", "Slightly fragrant", "None"])
        
        with col2:
            native_region = st.text_input("Native Region", placeholder="e.g., Southeast Asia")
            difficulty = st.selectbox("Growing Difficulty", ["", "Easy", "Moderate", "Difficult"])
            min_temp = st.number_input("Min Temperature (°C)", value=None, min_value=-10, max_value=50)
        
        with col3:
            max_temp = st.number_input("Max Temperature (°C)", value=None, min_value=-10, max_value=50)
            st.markdown("<br>", unsafe_allow_html=True)
            filter_btn = st.button("🎯 Apply Filters", type="primary", use_container_width=True)
        
        if filter_btn:
            with st.spinner("Filtering..."):
                try:
                    results = db.semantic_search(
                        genus=genus,
                        flower_color=flower_color,
                        native_region=native_region,
                        fragrance=fragrance,
                        difficulty=difficulty,
                        min_temp=min_temp,
                        max_temp=max_temp,
                        limit=50
                    )
                    
                    if results:
                        st.success(f"Found {len(results)} matching orchids")
                        
                        # Convert to DataFrame for better display
                        df = pd.DataFrame(results)
                        display_cols = ['Scientific_Name', 'Genus', 'Flower_Color', 'Temperature_Min_C', 
                                       'Temperature_Max_C', 'Native_Regions', 'Horticultural_Difficulty']
                        
                        # Filter only existing columns
                        available_cols = [col for col in display_cols if col in df.columns]
                        
                        st.dataframe(
                            df[available_cols].head(50),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Detailed view
                        st.markdown("---")
                        st.markdown("### Detailed Results")
                        
                        for i, result in enumerate(results[:10], 1):
                            with st.expander(f"{i}. {result.get('Scientific_Name', 'Unknown')}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Basic Information**")
                                    st.write(f"**Common Name:** {result.get('Common_Names', 'N/A')}")
                                    st.write(f"**Genus:** {result.get('Genus', 'N/A')}")
                                    st.write(f"**Family:** {result.get('Family', 'N/A')}")
                                    st.write(f"**Growth Habit:** {result.get('Growth_Habit', 'N/A')}")
                                
                                with col2:
                                    st.markdown("**Care Requirements**")
                                    st.write(f"**Difficulty:** {result.get('Horticultural_Difficulty', 'N/A')}")
                                    st.write(f"**Watering:** {result.get('Watering_Frequency', 'N/A')}")
                                    st.write(f"**Potting Media:** {result.get('Potting_Media', 'N/A')}")
                                    st.write(f"**Fertilizer:** {result.get('Fertilizer_Requirement', 'N/A')}")
                    else:
                        st.warning("No results match your filters. Try adjusting the criteria.")
                except Exception as e:
                    st.error(f"Filter error: {str(e)}")
                    st.info("💡 Try using fewer filters or broader criteria")
    
    # Tab 3: Combined Search
    with tab3:
        st.markdown("### 🔗 Combined Text + Filter Search")
        st.info("💡 Example: Search for 'fragrant' orchids in 'Southeast Asia' with 'Pink' flowers")
        
        # Text search input
        combined_text = st.text_input(
            "Text Search (optional)",
            placeholder="e.g., fragrant, beautiful, easy care",
            help="Search across multiple text fields"
        )
        
        st.markdown("#### Additional Filters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            c_genus = st.text_input("Genus", placeholder="e.g., Phalaenopsis", key="c_genus")
            c_color = st.text_input("Flower Color", placeholder="e.g., Pink", key="c_color")
        
        with col2:
            c_region = st.text_input("Native Region", placeholder="e.g., Southeast Asia", key="c_region")
            c_fragrance = st.text_input("Fragrance", placeholder="e.g., fragrant, sweet", key="c_fragrance")
        
        with col3:
            c_min_temp = st.number_input("Min Temp (°C)", value=None, min_value=-10, max_value=50, key="c_min")
            c_max_temp = st.number_input("Max Temp (°C)", value=None, min_value=-10, max_value=50, key="c_max")
        
        with col4:
            c_difficulty = st.selectbox("Difficulty", ["", "Easy", "Moderate", "Difficult"], key="c_diff")
        
        combined_btn = st.button("🔍 Search with Combined Criteria", type="primary", use_container_width=True)
        
        if combined_btn:
            with st.spinner("Searching..."):
                try:
                    results = db.combined_search(
                        text_query=combined_text if combined_text else None,
                        genus=c_genus,
                        flower_color=c_color,
                        native_region=c_region,
                        fragrance=c_fragrance,
                        min_temp=c_min_temp,
                        max_temp=c_max_temp,
                        difficulty=c_difficulty,
                        limit=50
                    )
                    
                    if results:
                        st.success(f"✅ Found {len(results)} matching orchids")
                        
                        # Summary stats
                        df = pd.DataFrame(results)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Results", len(results))
                        with col2:
                            unique_genera = df['Genus'].nunique() if 'Genus' in df.columns else 0
                            st.metric("Unique Genera", unique_genera)
                        with col3:
                            unique_colors = df['Flower_Color'].nunique() if 'Flower_Color' in df.columns else 0
                            st.metric("Colors Found", unique_colors)
                        with col4:
                            unique_regions = df['Native_Regions'].nunique() if 'Native_Regions' in df.columns else 0
                            st.metric("Regions", unique_regions)
                        
                        st.markdown("---")
                        
                        # Display results
                        for i, result in enumerate(results, 1):
                            with st.expander(f"**{i}. {result.get('Scientific_Name', 'Unknown')}** - {result.get('Common_Names', 'N/A')}"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown("**🌺 Flower Details**")
                                    st.write(f"**Color:** {result.get('Flower_Color', 'N/A')}")
                                    st.write(f"**Size:** {result.get('Flower_Size_cm', 'N/A')} cm")
                                    st.write(f"**Fragrance:** {result.get('Fragrance', 'N/A')}")
                                    if result.get('Fragrance_Description'):
                                        st.write(f"**Description:** {result['Fragrance_Description']}")
                                
                                with col2:
                                    st.markdown("**🌡️ Growing Conditions**")
                                    st.write(f"**Temperature:** {result.get('Temperature_Min_C', 'N/A')}-{result.get('Temperature_Max_C', 'N/A')}°C")
                                    st.write(f"**Humidity:** {result.get('Humidity_Min_Percent', 'N/A')}-{result.get('Humidity_Max_Percent', 'N/A')}%")
                                    st.write(f"**Difficulty:** {result.get('Horticultural_Difficulty', 'N/A')}")
                                
                                with col3:
                                    st.markdown("**🌍 Origin**")
                                    st.write(f"**Region:** {result.get('Native_Regions', 'N/A')}")
                                    st.write(f"**Habitat:** {result.get('Native_Habitat', 'N/A')}")
                                    st.write(f"**Climate:** {result.get('Climate_Type', 'N/A')}")
                                
                                if result.get('Special_Features'):
                                    st.success(f"✨ **Special:** {result['Special_Features']}")
                        
                        # Download option
                        st.markdown("---")
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="orchid_search_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ No results found. Try adjusting your search criteria:")
                        st.info("""
                        **Tips:**
                        - Use simpler/broader search terms
                        - Check spelling of region names
                        - Try searching with fewer filters
                        - Example that works: Color='Pink', Region='Southeast Asia'
                        """)
                except Exception as e:
                    st.error(f"❌ Search error: {str(e)}")
                    st.code(str(e))
    
    # Tab 4: Browse All
    with tab4:
        st.markdown("### Browse All Orchids")
        
        # Get all data
        all_results = db.semantic_search(limit=100)
        
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Column selector
            all_columns = df.columns.tolist()
            default_cols = ['Scientific_Name', 'Genus', 'Flower_Color', 'Native_Regions', 
                           'Temperature_Preference', 'Horticultural_Difficulty']
            
            selected_cols = st.multiselect(
                "Select columns to display",
                options=all_columns,
                default=default_cols
            )
            
            if selected_cols:
                st.dataframe(
                    df[selected_cols],
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
                
                # Download button
                csv = df[selected_cols].to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="orchid_data.csv",
                    mime="text/csv"
                )

else:
    st.info("👈 Please load the dataset from the sidebar to start searching!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🌸 Orchid Database powered by SQLite FTS5 | "
    "Data source: <a href='https://github.com/rashadul-se/orchids'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
