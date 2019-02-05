# Parts of this code were adapted from Sebastian Burgstaller-Muehlbacher's cdk_pywrapper
#  python pkg to make use of CDK in Python:
#   https://github.com/sebotic/cdk_pywrapper

import os
import py4j
import subprocess

from py4j.java_gateway import JavaGateway, GatewayParameters


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
            else:
                print("Compiling CDKBridge")
                command_str = "javac -cp {}:{}/ /mnt/code/modSAR/java/cdk_bridge.java"
                command_str = command_str.format(self.py4j_jar_path, self.cdk_jar_path)
                output = subprocess.run(command_str.split(), check=False, stdout=subprocess.PIPE)

                print("Starting CDKBridge")
                command_str = "java -cp {}:{}/:/mnt/code/modSAR/java/ CDKBridge &"
                command_str = command_str.format(self.py4j_jar_path, self.cdk_jar_path)
                subprocess.Popen(command_str.split(), stdout=subprocess.PIPE)
                self.is_server_running = True
                self.gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))

    def __del__(self):
        print("Cleaning up JavaGateway")
        self.gateway.shutdown()
        self.is_server_running = False
