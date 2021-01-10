#include "gtest/gtest.h"

#include "ioprint.h"
#include "exception.h"

using namespace core;

TEST(ioprint, int_test)
{
    const std::string ret = ioprint(23);
    ASSERT_EQ(ret, "23");
}

TEST(exception, requre1)
{
    try
    {
        REQUIRE(false, "Test msg: " << 1 << ", " << 2);
    }
    catch(const Exception& ex)
    {
        EXPECT_TRUE(std::string(ex.what()).find("Test msg: 1, 2") != std::string::npos) << ex.what();
    }
    catch (...)
    {
        FAIL() << "Wrong exception";
    }
}
