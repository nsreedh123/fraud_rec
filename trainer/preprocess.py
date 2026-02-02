import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

def load_dataset(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    logging.info("Loading transaction data...")
    train_transaction = pd.read_csv(os.path.join(data_path, 'train_transaction.csv'))
    test_transaction = pd.read_csv(os.path.join(data_path, 'test_transaction.csv'))

    # Load identity data
    logging.info("Loading identity data...")
    train_identity = pd.read_csv(os.path.join(data_path, 'train_identity.csv'))
    test_identity = pd.read_csv(os.path.join(data_path, 'test_identity.csv'))

    logging.info(f"\nTrain Transaction Shape: {train_transaction.shape}")
    logging.info(f"Test Transaction Shape: {test_transaction.shape}")
    logging.info(f"Train Identity Shape: {train_identity.shape}")
    logging.info(f"Test Identity Shape: {test_identity.shape}")
    
    return train_transaction, test_transaction, train_identity, test_identity

class FraudDataPreprocessor:
    @staticmethod
    def merge_datasets(transactions: pd.DataFrame, identities: pd.DataFrame) -> pd.DataFrame:
        logging.info("Merging transaction and identity data...")
        merged_data = transactions.merge(identities, on='TransactionID', how='left')
        logging.info(f"Merged Data Shape: {merged_data.shape}")
        return merged_data

    @staticmethod
    def reduce_mem_usage(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
        """Reduce memory usage by converting data types (safe version)"""
        start_mem = df.memory_usage(deep=True).sum() / 1024**2

        for col in df.columns:
            col_type = df[col].dtype

            # Skip non-numeric columns safely
            if not pd.api.types.is_numeric_dtype(col_type):
                continue

            # Coerce to numeric to avoid string contamination
            col_data = pd.to_numeric(df[col], errors="coerce")

            if col_data.isna().all():
                # Column is completely unusable for numeric downcasting
                continue

            c_min = col_data.min()
            c_max = col_data.max()

            # Integer types
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = col_data.astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = col_data.astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = col_data.astype(np.int32)
                else:
                    df[col] = col_data.astype(np.int64)

            # Float types
            else:
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = col_data.astype(np.float32)
                else:
                    df[col] = col_data.astype(np.float64)

        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        if verbose:
            logging.info(
                f"Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB "
                f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
            )

        return df
    
    @staticmethod
    def create_transaction_amount_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on transaction amount"""
        logging.info("Creating transaction amount features...")
        
        # Log transformation
        df['TransactionAmt_Log'] = np.log1p(df['TransactionAmt'])
        
        # Decimal part (cents) - often fraudsters use round numbers
        df['TransactionAmt_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int)
        
        # Is round amount (no cents)
        df['TransactionAmt_is_round'] = (df['TransactionAmt'] == df['TransactionAmt'].astype(int)).astype(int)
        
        # Amount bins
        df['TransactionAmt_bin'] = pd.cut(df['TransactionAmt'], 
                                        bins=[0, 50, 100, 200, 500, 1000, 5000, 10000, np.inf],
                                        labels=[0, 1, 2, 3, 4, 5, 6, 7]).astype(float)
        
        # Cents value
        df['TransactionAmt_cents'] = (df['TransactionAmt'] * 100 % 100).astype(int)
        
        # Common fraud amounts (ends in .00, .99, .95)
        df['TransactionAmt_ends_00'] = (df['TransactionAmt_cents'] == 0).astype(int)
        df['TransactionAmt_ends_99'] = (df['TransactionAmt_cents'] == 99).astype(int)
        df['TransactionAmt_ends_95'] = (df['TransactionAmt_cents'] == 95).astype(int)
        
        return df
    
    @staticmethod
    def create_time_features(df):
        """Create time-based features from TransactionDT"""
        logging.info("Creating time features...")
        
        # TransactionDT is seconds from a reference point
        # Convert to interpretable time units
        
        # Hour of day (0-23)
        df['Transaction_hour'] = np.floor(df['TransactionDT'] / 3600) % 24
        
        # Day of week (0-6)
        df['Transaction_dow'] = np.floor(df['TransactionDT'] / (3600 * 24)) % 7
        
        # Day of month approximation
        df['Transaction_dom'] = np.floor(df['TransactionDT'] / (3600 * 24)) % 30
        
        # Week number
        df['Transaction_week'] = np.floor(df['TransactionDT'] / (3600 * 24 * 7))
        
        # Is weekend
        df['Transaction_is_weekend'] = (df['Transaction_dow'] >= 5).astype(int)
        
        # Time of day categories
        df['Transaction_time_of_day'] = pd.cut(df['Transaction_hour'], 
                                                bins=[-1, 6, 12, 18, 24],
                                                labels=[0, 1, 2, 3]).astype(float)  # night, morning, afternoon, evening
        
        # Is night (00:00 - 06:00) - higher fraud risk
        df['Transaction_is_night'] = ((df['Transaction_hour'] >= 0) & (df['Transaction_hour'] < 6)).astype(int)
        
        # Is business hours (09:00 - 17:00)
        df['Transaction_is_business_hours'] = ((df['Transaction_hour'] >= 9) & (df['Transaction_hour'] < 17)).astype(int)
        
        return df
    
    @staticmethod
    def create_card_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on card information"""
        logging.info("Creating card features...")
        
        # Card combination features
        card_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
        
        for col in card_cols:
            if col in df.columns:
                # Fill NaN for string operations
                df[col] = df[col].fillna(-999)
        
        # Card1 + Card2 combination
        if 'card1' in df.columns and 'card2' in df.columns:
            df['card1_card2'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
        
        # Card type combinations
        if 'card4' in df.columns and 'card6' in df.columns:
            df['card4_card6'] = df['card4'].astype(str) + '_' + df['card6'].astype(str)
        
        # Card + Address combinations
        if 'card1' in df.columns and 'addr1' in df.columns:
            df['card1_addr1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
        
        if 'card1' in df.columns and 'addr2' in df.columns:
            df['card1_addr2'] = df['card1'].astype(str) + '_' + df['addr2'].astype(str)
        
        if 'card2' in df.columns and 'addr1' in df.columns:
            df['card2_addr1'] = df['card2'].astype(str) + '_' + df['addr1'].astype(str)
        
        # Card + Product combinations
        if 'card1' in df.columns and 'ProductCD' in df.columns:
            df['card1_ProductCD'] = df['card1'].astype(str) + '_' + df['ProductCD'].astype(str)
        
        if 'card2' in df.columns and 'ProductCD' in df.columns:
            df['card2_ProductCD'] = df['card2'].astype(str) + '_' + df['ProductCD'].astype(str)
        
        # Card + Time combinations
        if 'card1' in df.columns and 'Transaction_hour' in df.columns:
            df['card1_hour'] = df['card1'].astype(str) + '_' + df['Transaction_hour'].astype(int).astype(str)
        
        if 'card1' in df.columns and 'Transaction_dow' in df.columns:
            df['card1_dow'] = df['card1'].astype(str) + '_' + df['Transaction_dow'].astype(int).astype(str)
        
        return df
    
    @staticmethod
    def create_email_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on email domains"""
        logging.info("Creating email features...")
        
        # P_emaildomain features
        if 'P_emaildomain' in df.columns:
            # Extract domain prefix
            df['P_emaildomain_prefix'] = df['P_emaildomain'].apply(
                lambda x: str(x).split('.')[0] if pd.notna(x) else 'unknown'
            )
            
            # Extract domain suffix (com, net, org, etc.)
            df['P_emaildomain_suffix'] = df['P_emaildomain'].apply(
                lambda x: str(x).split('.')[-1] if pd.notna(x) and '.' in str(x) else 'unknown'
            )
            
            # Is common email provider
            common_providers = ['gmail', 'yahoo', 'hotmail', 'outlook', 'aol', 'icloud', 'live', 'msn']
            df['P_email_is_common'] = df['P_emaildomain_prefix'].apply(
                lambda x: 1 if x.lower() in common_providers else 0
            )
            
            # Is business email (not common provider)
            df['P_email_is_business'] = (df['P_email_is_common'] == 0).astype(int)
            
            # Email domain length
            df['P_emaildomain_len'] = df['P_emaildomain'].apply(
                lambda x: len(str(x)) if pd.notna(x) else 0
            )
        
        # R_emaildomain features
        if 'R_emaildomain' in df.columns:
            df['R_emaildomain_prefix'] = df['R_emaildomain'].apply(
                lambda x: str(x).split('.')[0] if pd.notna(x) else 'unknown'
            )
            
            df['R_emaildomain_suffix'] = df['R_emaildomain'].apply(
                lambda x: str(x).split('.')[-1] if pd.notna(x) and '.' in str(x) else 'unknown'
            )
            
            df['R_email_is_common'] = df['R_emaildomain_prefix'].apply(
                lambda x: 1 if x.lower() in ['gmail', 'yahoo', 'hotmail', 'outlook', 'aol', 'icloud', 'live', 'msn'] else 0
            )
        
        # Email match features
        if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
            # Same email domain
            df['email_domain_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)
            
            # Same email prefix
            df['email_prefix_match'] = (df['P_emaildomain_prefix'] == df['R_emaildomain_prefix']).astype(int)
            
            # Both emails missing
            df['both_emails_missing'] = (df['P_emaildomain'].isna() & df['R_emaildomain'].isna()).astype(int)
            
            # Only P email present
            df['only_P_email'] = (df['P_emaildomain'].notna() & df['R_emaildomain'].isna()).astype(int)
            
            # Only R email present
            df['only_R_email'] = (df['P_emaildomain'].isna() & df['R_emaildomain'].notna()).astype(int)
        
        return df
    
    @staticmethod
    def create_device_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on device information"""
        logging.info("Creating device features...")
        
        # DeviceType
        if 'DeviceType' in df.columns:
            df['DeviceType_is_mobile'] = (df['DeviceType'] == 'mobile').astype(int)
            df['DeviceType_is_desktop'] = (df['DeviceType'] == 'desktop').astype(int)
        
        # DeviceInfo
        if 'DeviceInfo' in df.columns:
            # Extract device brand
            df['Device_brand'] = df['DeviceInfo'].apply(
                lambda x: str(x).split('/')[0].split()[0] if pd.notna(x) else 'unknown'
            )
            
            # Device info length (complexity indicator)
            df['DeviceInfo_len'] = df['DeviceInfo'].apply(
                lambda x: len(str(x)) if pd.notna(x) else 0
            )
            
            # Is specific device types
            df['Device_is_Samsung'] = df['DeviceInfo'].apply(
                lambda x: 1 if pd.notna(x) and 'samsung' in str(x).lower() else 0
            )
            df['Device_is_iOS'] = df['DeviceInfo'].apply(
                lambda x: 1 if pd.notna(x) and ('ios' in str(x).lower() or 'iphone' in str(x).lower() or 'ipad' in str(x).lower()) else 0
            )
            df['Device_is_Windows'] = df['DeviceInfo'].apply(
                lambda x: 1 if pd.notna(x) and 'windows' in str(x).lower() else 0
            )
        
        # Browser features from id_31
        if 'id_31' in df.columns:
            df['Browser'] = df['id_31'].apply(
                lambda x: str(x).split()[0].lower() if pd.notna(x) else 'unknown'
            )
            
            df['Browser_is_chrome'] = df['id_31'].apply(
                lambda x: 1 if pd.notna(x) and 'chrome' in str(x).lower() else 0
            )
            df['Browser_is_safari'] = df['id_31'].apply(
                lambda x: 1 if pd.notna(x) and 'safari' in str(x).lower() else 0
            )
            df['Browser_is_firefox'] = df['id_31'].apply(
                lambda x: 1 if pd.notna(x) and 'firefox' in str(x).lower() else 0
            )
            df['Browser_is_edge'] = df['id_31'].apply(
                lambda x: 1 if pd.notna(x) and 'edge' in str(x).lower() else 0
            )
        
        # OS features from id_30
        if 'id_30' in df.columns:
            df['OS'] = df['id_30'].apply(
                lambda x: str(x).split()[0].lower() if pd.notna(x) else 'unknown'
            )
            
            df['OS_is_Windows'] = df['id_30'].apply(
                lambda x: 1 if pd.notna(x) and 'windows' in str(x).lower() else 0
            )
            df['OS_is_Mac'] = df['id_30'].apply(
                lambda x: 1 if pd.notna(x) and 'mac' in str(x).lower() else 0
            )
            df['OS_is_iOS'] = df['id_30'].apply(
                lambda x: 1 if pd.notna(x) and 'ios' in str(x).lower() else 0
            )
            df['OS_is_Android'] = df['id_30'].apply(
                lambda x: 1 if pd.notna(x) and 'android' in str(x).lower() else 0
            )
        
        # Screen resolution from id_33
        if 'id_33' in df.columns:
            df['Screen_width'] = df['id_33'].apply(
                lambda x: int(str(x).split('x')[0]) if pd.notna(x) and 'x' in str(x) else -1
            )
            df['Screen_height'] = df['id_33'].apply(
                lambda x: int(str(x).split('x')[1]) if pd.notna(x) and 'x' in str(x) else -1
            )
            df['Screen_area'] = df['Screen_width'] * df['Screen_height']
            df['Screen_aspect_ratio'] = df.apply(
                lambda row: row['Screen_width'] / row['Screen_height'] if row['Screen_height'] > 0 else -1, axis=1
            )
        
        return df
    
    @staticmethod
    def create_address_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on address information"""
        logging.info("Creating address features...")
        
        # Address combinations
        if 'addr1' in df.columns and 'addr2' in df.columns:
            df['addr1_addr2'] = df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
        
        # Address + ProductCD
        if 'addr1' in df.columns and 'ProductCD' in df.columns:
            df['addr1_ProductCD'] = df['addr1'].astype(str) + '_' + df['ProductCD'].astype(str)
        
        # Address distance from P_emaildomain (proxy for geographic mismatch)
        if 'addr1' in df.columns:
            df['addr1_missing'] = df['addr1'].isna().astype(int)
        
        if 'addr2' in df.columns:
            df['addr2_missing'] = df['addr2'].isna().astype(int)
        
        # Both addresses missing
        if 'addr1' in df.columns and 'addr2' in df.columns:
            df['both_addr_missing'] = (df['addr1'].isna() & df['addr2'].isna()).astype(int)
        
        # dist1 and dist2 features
        if 'dist1' in df.columns:
            df['dist1_missing'] = df['dist1'].isna().astype(int)
            df['dist1_log'] = np.log1p(df['dist1'].fillna(0))
        
        if 'dist2' in df.columns:
            df['dist2_missing'] = df['dist2'].isna().astype(int)
            df['dist2_log'] = np.log1p(df['dist2'].fillna(0))
        
        return df
    
    def create_v_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features from V columns"""
        logging.info("Creating V-column aggregation features...")
        
        # Get all V columns
        v_cols = [col for col in df.columns if col.startswith('V')]
        
        if len(v_cols) == 0:
            return df
        
        # Group V columns by their correlation patterns (based on EDA from competition)
        v_groups = {
            'v1': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11'],
            'v2': ['V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26'],
            'v3': ['V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34'],
            'v4': ['V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52'],
            'v5': ['V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74'],
            'v6': ['V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94'],
            'v7': ['V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137'],
        }
        
        for group_name, group_cols in v_groups.items():
            # Filter to existing columns
            existing_cols = [col for col in group_cols if col in df.columns]
            
            if len(existing_cols) > 0:
                # Sum of group
                df[f'{group_name}_sum'] = df[existing_cols].sum(axis=1)
                
                # Mean of group
                df[f'{group_name}_mean'] = df[existing_cols].mean(axis=1)
                
                # Std of group
                df[f'{group_name}_std'] = df[existing_cols].std(axis=1)
                
                # NaN count in group
                df[f'{group_name}_nan_count'] = df[existing_cols].isna().sum(axis=1)
        
        # Overall V statistics
        existing_v_cols = [col for col in v_cols if col in df.columns]
        if len(existing_v_cols) > 0:
            df['V_sum_all'] = df[existing_v_cols].sum(axis=1)
            df['V_mean_all'] = df[existing_v_cols].mean(axis=1)
            df['V_std_all'] = df[existing_v_cols].std(axis=1)
            df['V_nan_count_all'] = df[existing_v_cols].isna().sum(axis=1)
            df['V_nan_ratio'] = df['V_nan_count_all'] / len(existing_v_cols)
        
        return df
    
    def create_c_d_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features from C (count) and D (timedelta) columns"""
        logging.info("Creating C and D column features...")
        
        # C columns (counting features)
        c_cols = [col for col in df.columns if col.startswith('C') and col[1:].isdigit()]
        
        if len(c_cols) > 0:
            existing_c_cols = [col for col in c_cols if col in df.columns]
            
            # Aggregations
            df['C_sum'] = df[existing_c_cols].sum(axis=1)
            df['C_mean'] = df[existing_c_cols].mean(axis=1)
            df['C_std'] = df[existing_c_cols].std(axis=1)
            df['C_max'] = df[existing_c_cols].max(axis=1)
            df['C_min'] = df[existing_c_cols].min(axis=1)
            
            # Log transformations for key C columns
            for col in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']:
                if col in df.columns:
                    df[f'{col}_log'] = np.log1p(df[col].fillna(0))
        
        # D columns (timedelta features)
        d_cols = [col for col in df.columns if col.startswith('D') and col[1:].isdigit()]
        
        if len(d_cols) > 0:
            existing_d_cols = [col for col in d_cols if col in df.columns]
            
            # Aggregations
            df['D_sum'] = df[existing_d_cols].sum(axis=1)
            df['D_mean'] = df[existing_d_cols].mean(axis=1)
            df['D_std'] = df[existing_d_cols].std(axis=1)
            df['D_nan_count'] = df[existing_d_cols].isna().sum(axis=1)
            
            # D1 is often important (days since card was first used)
            if 'D1' in df.columns:
                df['D1_missing'] = df['D1'].isna().astype(int)
                df['D1_log'] = np.log1p(df['D1'].fillna(0))
            
            # D15 is also important
            if 'D15' in df.columns:
                df['D15_missing'] = df['D15'].isna().astype(int)
                df['D15_log'] = np.log1p(df['D15'].fillna(0))
        
        return df
    
    @staticmethod
    def create_m_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features from M (match) columns"""
        logging.info("Creating M column features...")
        
        m_cols = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
        existing_m_cols = [col for col in m_cols if col in df.columns]
        
        if len(existing_m_cols) > 0:
            # Count of True values
            for col in existing_m_cols:
                df[f'{col}_isT'] = (df[col] == 'T').astype(int)
                df[f'{col}_isF'] = (df[col] == 'F').astype(int)
                df[f'{col}_missing'] = df[col].isna().astype(int)
            
            # Total True count
            t_cols = [f'{col}_isT' for col in existing_m_cols]
            df['M_true_count'] = df[t_cols].sum(axis=1)
            
            # Total False count
            f_cols = [f'{col}_isF' for col in existing_m_cols]
            df['M_false_count'] = df[f_cols].sum(axis=1)
            
            # Total missing count
            m_missing_cols = [f'{col}_missing' for col in existing_m_cols]
            df['M_missing_count'] = df[m_missing_cols].sum(axis=1)
            
            # Ratio of True to total non-missing
            df['M_true_ratio'] = df['M_true_count'] / (df['M_true_count'] + df['M_false_count'] + 0.001)
        
        return df
    
    @staticmethod
    def create_id_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features from identity columns"""
        logging.info("Creating ID features...")
        
        # id_01 to id_11 are numerical
        id_num_cols = [f'id_0{i}' for i in range(1, 10)] + ['id_10', 'id_11']
        existing_id_num = [col for col in id_num_cols if col in df.columns]
        
        if len(existing_id_num) > 0:
            df['id_num_nan_count'] = df[existing_id_num].isna().sum(axis=1)
            df['id_num_mean'] = df[existing_id_num].mean(axis=1)
            df['id_num_std'] = df[existing_id_num].std(axis=1)
        
        # id_12 to id_38 are categorical
        # Check specific important ones
        if 'id_12' in df.columns:
            df['id_12_isFound'] = (df['id_12'] == 'Found').astype(int)
        
        if 'id_15' in df.columns:
            df['id_15_isNew'] = (df['id_15'] == 'New').astype(int)
            df['id_15_isFound'] = (df['id_15'] == 'Found').astype(int)
        
        if 'id_16' in df.columns:
            df['id_16_isFound'] = (df['id_16'] == 'Found').astype(int)
        
        if 'id_28' in df.columns:
            df['id_28_isNew'] = (df['id_28'] == 'New').astype(int)
            df['id_28_isFound'] = (df['id_28'] == 'Found').astype(int)
        
        if 'id_29' in df.columns:
            df['id_29_isFound'] = (df['id_29'] == 'Found').astype(int)
        
        # id_34 (match status)
        if 'id_34' in df.columns:
            df['id_34_match'] = df['id_34'].apply(
                lambda x: int(str(x).split(':')[1]) if pd.notna(x) and ':' in str(x) else -1
            )
        
        # id_36 features
        if 'id_36' in df.columns:
            df['id_36_isT'] = (df['id_36'] == 'T').astype(int)
        
        # id_37, id_38 features
        if 'id_37' in df.columns:
            df['id_37_isT'] = (df['id_37'] == 'T').astype(int)
        
        if 'id_38' in df.columns:
            df['id_38_isT'] = (df['id_38'] == 'T').astype(int)
        
        return df
    
    @staticmethod
    def create_aggregation_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create frequency and aggregation features"""
        logging.info("Creating aggregation features...")
        
        # Combine for consistent encoding
        train_df['is_train'] = 1
        test_df['is_train'] = 0
        combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        
        # Columns for frequency encoding
        freq_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
                    'ProductCD', 'DeviceType', 'DeviceInfo']
        
        for col in freq_cols:
            if col in combined.columns:
                # Frequency encoding
                freq = combined[col].value_counts().to_dict()
                combined[f'{col}_freq'] = combined[col].map(freq)
        
        # Transaction amount aggregations by card
        agg_cols = ['card1', 'card2', 'card1_card2', 'addr1']
        
        for col in agg_cols:
            if col in combined.columns:
                # Mean transaction amount
                agg = combined.groupby(col)['TransactionAmt'].mean().to_dict()
                combined[f'{col}_TransactionAmt_mean'] = combined[col].map(agg)
                
                # Std transaction amount
                agg = combined.groupby(col)['TransactionAmt'].std().to_dict()
                combined[f'{col}_TransactionAmt_std'] = combined[col].map(agg)
                
                # Transaction amount deviation from mean
                combined[f'{col}_TransactionAmt_dev'] = combined['TransactionAmt'] - combined[f'{col}_TransactionAmt_mean']
        
        # Split back
        train_df = combined[combined['is_train'] == 1].drop('is_train', axis=1)
        test_df = combined[combined['is_train'] == 0].drop('is_train', axis=1)
        
        return train_df, test_df
    
    @staticmethod
    def pre_process(df_tx: pd.DataFrame, df_id: pd.DataFrame) -> pd.DataFrame:
        """Run all preprocessing steps"""
        df = FraudDataPreprocessor.merge_datasets(df_tx, df_id)
        df = FraudDataPreprocessor.create_transaction_amount_features(df)
        df = FraudDataPreprocessor.create_time_features(df)
        df = FraudDataPreprocessor.create_card_features(df)
        df = FraudDataPreprocessor.create_email_features(df)
        df = FraudDataPreprocessor.create_device_features(df)
        df = FraudDataPreprocessor.create_address_features(df)
        df = FraudDataPreprocessor.create_v_features(df)
        df = FraudDataPreprocessor.create_c_d_features(df)
        df = FraudDataPreprocessor.create_m_features(df)
        df = FraudDataPreprocessor.create_id_features(df)
        
        return df
    
    @staticmethod
    def fit_transform(train_df: pd.DataFrame, train_id: pd.DataFrame, test_df: pd.DataFrame, test_id: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process datasets and reduce memory usage"""
        logging.info("Preprocessing training data...")
        train_df = FraudDataPreprocessor.pre_process(train_df, train_id)
        
        logging.info("Preprocessing test data...")
        test_df = FraudDataPreprocessor.pre_process(test_df, test_id)
        
        logging.info("Creating aggregation features...")
        train_df, test_df = FraudDataPreprocessor.create_aggregation_features(train_df, test_df)
        
        logging.info("Reducing memory usage for training data...")
        train_df = FraudDataPreprocessor.reduce_mem_usage(train_df)
        
        logging.info("Reducing memory usage for test data...")
        test_df = FraudDataPreprocessor.reduce_mem_usage(test_df)
        
        return train_df, test_df