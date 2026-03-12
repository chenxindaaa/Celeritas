#ifndef __SINGLETON_H__
#define __SINGLETON_H__

template<typename T>
class Singleton {
public:
    static T& getInstance() {
        static T instance; 
        return instance;
    }

    virtual ~Singleton() = default;
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
protected:
    Singleton() = default;
};

#endif //__UTILS_HPP__//
