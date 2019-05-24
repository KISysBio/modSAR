import os
import py4j
import time
import subprocess

import numpy as np
import pandas as pd

from py4j.java_gateway import JavaGateway, GatewayParameters

from .utils import print_progress_bar


class JavaCDKBridge:
    """Class responsible for establishing the bridge between Python and the Java library CDK


    Usage:
        `cdk_bridge = JavaCDKBridge()`
        `cdk_bridge.start_cdk_java_bridge()`

        Then to use Java objects,
            use the py4j's JavaGateway object that is created in the cdk_bridge object:

            `java_gateway = cdk_bridge.gateway`

            For example:

            `java_gateway.jvm.java.io.StringWriter()`

    """

    def __init__(self, py4j_jar_path=None, cdk_jar_path=None):
        """Initialise paths"""

        if py4j_jar_path is None:
            self.py4j_jar_path = py4j.java_gateway.find_jar_path()
        else:
            self.py4j_jar_path = py4j_jar_path

        if cdk_jar_path is None:
            jar_path = os.getenv('JAR_PATH')
            cdk_version = os.getenv('CDK_VERSION')
            self.cdk_jar_path = "%s/cdk-%s.jar" % (jar_path, cdk_version)
        else:
            self.cdk_jar_path = cdk_jar_path

        self.is_server_running = False
        self.gateway = None

    def start_cdk_java_bridge(self):
        """Start a JVM bridge so we can access CDK capabilities from within Python"""
        with subprocess.Popen(['ps aux | grep CDK'], shell=True, stdout=subprocess.PIPE) as proc:
            line = proc.stdout.read()
            if 'CDKBridge' in str(line):
                print('CDK Bridge process running')
                self.is_server_running = True
                if self.gateway is None:
                    self.gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
            else:
                # print("Compiling CDKBridge")
                # command_str = "javac -cp {}:{}/ /mnt/code/modSAR/java/cdk_bridge.java"
                # command_str = command_str.format(self.py4j_jar_path, self.cdk_jar_path)
                # output = subprocess.run(command_str.split(), check=False, stdout=subprocess.PIPE)

                print("Starting CDKBridge")
                command_str = "java -cp {}:{}/:/mnt/code/modSAR/java/ CDKBridge &"
                command_str = command_str.format(self.py4j_jar_path, self.cdk_jar_path)
                subprocess.Popen(command_str.split(), stdout=subprocess.PIPE)
                self.is_server_running = True
                self.gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
        time.sleep(4)

    def __del__(self):
        print("Cleaning up JavaGateway")
        self.gateway.shutdown()
        self.is_server_running = False


class CDKUtils:
    """"""

    def __init__(self):
        self.java_bridge = JavaCDKBridge()
        self.java_bridge.start_cdk_java_bridge()

        self.cdk = self.java_bridge.gateway.new_jvm_view().org.openscience.cdk
        self.builder = self.cdk.DefaultChemObjectBuilder.getInstance()
        self.smiles_parser = self.cdk.smiles.SmilesParser(self.builder)
        self.descriptors_list = self._get_descriptors_list()

    def _get_descriptors_list(self, descriptor_file='modSAR/descriptors_list.csv'):
        descriptors_list = pd.read_csv(descriptor_file)

        def remove_prefix(java_class_name):
            return java_class_name.replace('org.openscience.cdk', 'self.cdk') + '()'

        descriptors_list['object_invocation'] = descriptors_list['descriptorClass'].apply(remove_prefix)
        return descriptors_list

    def get_descriptor_values(self, mol, descriptor):
        descriptor_names = [desc_name for desc_name in descriptor.getDescriptorNames()]
        descriptor_vals = descriptor.calculate(mol).getValue().toString().split(',')
        desc_values = pd.Series({descriptor_names[j]: descriptor_vals[j]
                                 for j in range(len(descriptor_names))})
        return desc_values

    def calculate_descriptors(self, df, smiles_column):
        """
        Calculate all descriptors for each entry in the DataFrame

        Args:
            df (DataFrame):      Bioactivities DataFrame
            smiles_column (str): DataFrame column that contains SMILES
        """

        # TODO: Calculate descriptors in parallel

        cdk_molecules = df[smiles_column].apply(lambda smiles: self.smiles_parser.parseSmiles(smiles))

        num_descriptor_classes = len(self.descriptors_list)
        descriptor_values = []
        for i, descriptor_invocation in enumerate(self.descriptors_list['object_invocation']):
            descriptor = eval(descriptor_invocation)
            descriptor.initialise(self.builder)

            print("(%02d/%02d) Collecting descriptors from class %s" %
                  (i + 1, num_descriptor_classes, descriptor_invocation))
            values = cdk_molecules.apply(lambda mol: self.get_descriptor_values(mol, descriptor))
            descriptor_values.append(values)
        descriptors_df = pd.concat(descriptor_values, axis=1)

        # Remove empty columns
        is_empty_column = descriptors_df.apply(lambda x: sum(x == 'NaN'), axis=0) == descriptors_df.shape[0]
        descriptors_df = descriptors_df.loc[:, ~is_empty_column].copy()

        for col in descriptors_df.columns:
            descriptors_df[col] = descriptors_df[col].astype(float)
        return descriptors_df

    def calculate_fingerprint(self, smiles, circular_type=3):
        """Calculate circular fingerprint

        Args:
            smiles (str): SMILES representing a molecule
            circular_type (int): Type of circular fingerprint.
                ECFP0 = 1, ECFP2 = 2, ECFP4 = 3, ECFP6 = 4,
                FCFP0 = 5, FCFP2 = 6, FCFP4 = 7, FCFP6 = 8
        """

        fingerprinter = self.cdk.fingerprint.CircularFingerprinter(circular_type)
        mol = self.smiles_parser.parseSmiles(smiles)
        return fingerprinter.getBitFingerprint(mol)

    def calculate_tanimoto_similarity(self, smiles1, smiles2):
        """Calculate Tanimoto similarity between two molecules

        Args:
            smiles1 (str): SMILES string representing a first molecule
            smiles2 (str): SMILES string representing a second molecule
        """
        fp1 = self.calculate_fingerprint(smiles1)
        fp2 = self.calculate_fingerprint(smiles2)
        sim = self.cdk.similarity.Tanimoto.calculate(fp1, fp2)
        return sim

    def calculate_pairwise_tanimoto(self, df, smiles_column, circular_type=3):
        """

        Args:
            circular_type (int): Type of circular fingerprint.
                ECFP0 = 1, ECFP2 = 2, ECFP4 = 3, ECFP6 = 4,
                FCFP0 = 5, FCFP2 = 6, FCFP4 = 7, FCFP6 = 8
        """
        num_samples = len(df)
        total_comparisons = num_samples * (num_samples - 1)

        count = 0
        print_progress_bar(count, total_comparisons)
        matrix = np.zeros((num_samples, num_samples), dtype="f8")
        for i in range(num_samples):
            mol1 = df[smiles_column].iloc[i]
            for j in range(num_samples):
                count += 1
                if j < i:
                    print_progress_bar(count, total_comparisons)
                    continue
                elif j == i:
                    matrix[i, j] = 0
                else:
                    mol2 = df[smiles_column].iloc[j]

                    sim = self.calculate_tanimoto_similarity(mol1, mol2)
                    matrix[i, j] = sim
                    matrix[j, i] = sim
                print_progress_bar(count, total_comparisons)
        return matrix
