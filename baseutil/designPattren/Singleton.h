#pragma once

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