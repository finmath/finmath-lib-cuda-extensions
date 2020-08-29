# 
# This script performs a release for multiple profiles
#
# Before running this script you should check the profiles via
# 	mvn test -Dcuda.version=10.0
# and
# 	mvn test
#
# To create a new release you need to run
# 	mvn release:prepare
# before running this script and check if this is successful.
#
# To re-release a given tag / version you need to run
# 	git checkout finmath-lib-<verion>
# before running this script.
#
# If this script is run on a newly setup system, the following things need to be setup:
# - .m2/settings.xml - has to contain the passwords for sonatype, github and finmath.net site
# - .keyring has to contain the gpg keys
# - a gpg app has be installed (referenced in the .m2/settings.mxl)
# - you have to run mvn site:deploy manually once to get prompted for adding the RSA fingerprint
#

set -o verbose

#
# Run mvn release:prepare manually
# to check if the none of the tests fails (an additional run of prepare is not harmful).
#
 	mvn release:prepare
#

# relase the default profile
echo ###################
echo # RELEASE:PERFORM #
echo ###################

mvn release:perform

# deploy the other profiles (we do this skipping tests)
cd target/checkout/
mvn verify javadoc:jar source:jar gpg:sign deploy:deploy -Dcuda.version=6.0 -DskipTests=true
mvn verify javadoc:jar source:jar gpg:sign deploy:deploy -Dcuda.version=8.0 -DskipTests=true
mvn verify javadoc:jar source:jar gpg:sign deploy:deploy -Dcuda.version=9.2 -DskipTests=true
mvn verify javadoc:jar source:jar gpg:sign deploy:deploy -Dcuda.version=10.1 -DskipTests=true

# deploy site (clover:instrument takes a long time)
mvn clover:instrument site site:stage site-deploy
